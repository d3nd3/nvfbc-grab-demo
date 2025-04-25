#define GL_GLEXT_PROTOTYPES // Required for GL extension prototypes (e.g. glEGLImageTargetTexture2DOES)
#define GLX_GLXEXT_PROTOTYPES // Required for GLX extension prototypes (e.g. glXCreateContextAttribsARB)

#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <GL/gl.h> // Core OpenGL functions (GL 1.1+)
#include <GL/glext.h> // OpenGL extensions (GL 1.2+ core, plus vendor extensions)
#include <GL/glcorearb.h> // Modern OpenGL core profile functions (defines PFN types)


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h> // For sleep
#include <errno.h> // For strerror

#include <pthread.h>
#include <atomic>
#include <mutex>
#include <vector>
#include <stdexcept>

#include <dlfcn.h> // For dynamic loading

// Helper functions for dynamic library loading
static void *os_dlopen(const char *name) {
    void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", name, dlerror());
    }
    return handle;
}
static void *os_dlsym(void *handle, const char *name) {
    // Clear existing error
    dlerror();
    void *symbol = dlsym(handle, name);
    // Check for error if symbol is NULL, but remember NULL can be a valid symbol value
    // For function pointers, NULL is usually an error.
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
         fprintf(stderr, "dlsym(%s) failed: %s\n", name, dlerror());
    }
    return symbol;
}
static int os_dlclose(void *handle) {
    int ret = dlclose(handle);
    if (ret != 0) {
        fprintf(stderr, "dlclose failed: %s\n", dlerror());
    }
    return ret;
}

// Assume NvFBC.h is available in the include path
// Include NvFBC.h to get necessary types and struct definitions
#include "NvFBC.h"

// --- Manual Defines for potentially missing NvFBC constants ---
// These might be needed if your NvFBC.h or build environment doesn't expose them correctly.
// Verify these values against your NvFBC.h if possible.
#ifndef NVFBC_TOGL_MAX_BUFFERS
#define NVFBC_TOGL_MAX_BUFFERS 4 // Common value, adjust if your NvFBC.h is different
#endif
#ifndef NVFBC_ERROR_GL_TIMEOUT
#define NVFBC_ERROR_GL_TIMEOUT 21 // Common value (NVFBC_STATUS enumeration), adjust if different
#endif
#ifndef NVFBC_ERROR_GL_PRECONDITION_NOT_MET
#define NVFBC_ERROR_GL_PRECONDITION_NOT_MET 23 // Common value (NVFBC_STATUS enumeration), adjust if different
#endif


// Define function pointer type for NvFBCCreateInstance
typedef NVFBCSTATUS (*PFNNVFBCCreateInstance)(NVFBC_API_FUNCTION_LIST *pFunctions);

// Function pointers for GLX extensions (loaded via glXGetProcAddress)
static PFNGLXMAKECONTEXTCURRENTPROC p_glXMakeContextCurrent = NULL;
static PFNGLXCREATECONTEXTATTRIBSARBPROC p_glXCreateContextAttribsARB = NULL;
static PFNGLXCREATEPBUFFERPROC p_glXCreatePbuffer = NULL;
static PFNGLXCHOOSEFBCONFIGPROC p_glXChooseFBConfig = NULL;
// glXSwapBuffers is a core GLX function, called directly.

// Minimal Shader Source (passthrough vertex, texture fragment)
// Note: NvFBC textures might need Y-flipping depending on driver/NvFBC version and how they map to GL textures.
// The current texture coordinates are set up assuming 0,0 is bottom-left in the texture and maps to the bottom-left of the quad.
// If your capture appears upside down, flip the V coordinates: change 0.0/1.0 to 1.0/0.0 and 1.0/0.0 to 0.0/1.0 in the vertices array.
const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D screenTexture; // Sampler uniform
    void main() {
        FragColor = texture(screenTexture, TexCoord);
    }
)glsl";


// --- Shared Data Structure ---
struct shared_data {
    std::atomic<bool> stop_threads = false;
    std::atomic<bool> recreate_session = true; // Simulate initial setup/update

    // NvFBC library handle and function list (Main Thread loads, GLX Thread uses)
    void *nvfbc_lib_handle = NULL; // Loaded by main thread
    NVFBC_API_FUNCTION_LIST nvFBC_api = {.dwVersion = NVFBC_VERSION}; // This will hold pointers from dlsym

    // NVFBC Session (Accessed by Capture Thread, protected by mutex for create/destroy)
    NVFBC_SESSION_HANDLE nvfbc_session = -1;
    NVFBC_TOGL_SETUP_PARAMS togl_setup_params; // Protected by session_state_mutex (specifically the dwTextures field)
    std::atomic<bool> has_capture_session = false; // Protected by session_state_mutex for modification
    pthread_mutex_t session_state_mutex; // Protects nvfbc_session, has_capture_session, and fields of togl_setup_params modified by session setup

    // GLX Capture Thread handles and state (Owned by GLX Thread)
    Display *glx_dpy = NULL;        // GLX thread's X11 display connection
    Window glx_win = 0;             // GLX thread's X11 window
    GLXContext glx_ctx = NULL;      // GLX context for NvFBC/rendering ops
    GLXPbuffer glx_pbuffer = None;  // Dummy Pbuffer for making GLX context current for NvFBC calls
    GLXFBConfig glx_fb_config = NULL; // FBConfig used for context/drawables
    Atom wm_delete_window = None;   // Atom for window close event

    // Frame dimensions (updated by GLX thread)
    std::atomic<uint32_t> frame_width = 0;
    std::atomic<uint32_t> frame_height = 0;

    // Rendering state for GLX Thread (Owned by GLX Thread)
    GLuint current_grabbed_texture = 0; // Texture ID from NvFBC grab
    GLuint shader_program = 0;
    GLuint quad_vao = 0; // Minimal quad VAO
    GLuint quad_vbo = 0; // Minimal quad VBO
    GLint texture_sampler_location = -1; // Location of the sampler uniform

    // Settings (minimal, hardcoded or read once)
    int screen_id = -1; // -1 for entire desktop
    int fps = 60; // Target FPS
};

static struct shared_data data;

// Helper to compile shaders and link program (using direct GL calls)
GLuint createShaderProgram(const char* vShaderCode, const char* fShaderCode) {
    GLuint vertex, fragment, program;

    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
        glDeleteShader(vertex);
        return 0;
    }

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return 0;
    }

    // Shader Program
    program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glGetProgramiv(program,GL_LINK_STATUS, &success); // Use glGetProgramiv for linking status
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        glDeleteProgram(program);
        return 0;
    }

    // Delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    fprintf(stdout, "Shader program created successfully (%u).\n", program);
    return program;
}

// Shared global pointer for NvFBCCreateInstance loaded by main thread
static PFNNVFBCCreateInstance s_NvFBCCreateInstance_ptr = NULL;
// --- GLX Capture and Render Thread Function ---
static void *glx_capture_render_thread(void *param)
{
    fprintf(stdout, "GLX Capture/Render Thread: Started.\n");

    bool setup_success = false; // Flag to track overall setup success
    NVFBC_API_FUNCTION_LIST* nvFBC = &data.nvFBC_api; // Pointer to the shared API struct

    do { // Use a do-while(false) loop to allow using break for early exit during setup

        // --- 1. Open X11 Display and create GLX context/Window/Pbuffer ---
        data.glx_dpy = XOpenDisplay(NULL);
        if (!data.glx_dpy) {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to open X11 display.\n");
            break; // Exit setup block
        }
        int screen = DefaultScreen(data.glx_dpy);
        Window root_win = XRootWindow(data.glx_dpy, screen);

        // Get GLX extension function pointers needed for setup
        // NOTE: glXGetProcAddress requires a context to potentially be current on *some* drawable.
        // While not strictly guaranteed by standard, drivers often allow querying before
        // context creation if glXQueryExtensionsString or similar are available.
        // However, querying *after* creating a context is more reliable.
        // Let's defer getting these until after context creation, but store them here.
        PFNGLXMAKECONTEXTCURRENTPROC local_glXMakeContextCurrent = (PFNGLXMAKECONTEXTCURRENTPROC)glXGetProcAddress((const GLubyte*)"glXMakeContextCurrent");
        PFNGLXCREATECONTEXTATTRIBSARBPROC local_glXCreateContextAttribsARB = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
        PFNGLXCREATEPBUFFERPROC local_glXCreatePbuffer = (PFNGLXCREATEPBUFFERPROC)glXGetProcAddress((const GLubyte*)"glXCreatePbuffer");
        PFNGLXCHOOSEFBCONFIGPROC local_glXChooseFBConfig = (PFNGLXCHOOSEFBCONFIGPROC)glXGetProcAddress((const GLubyte*)"glXChooseFBConfig");

        if (!local_glXChooseFBConfig) { fprintf(stderr, "GLX Capture/Render Thread: glXChooseFBConfig not found. Cannot select FBConfig. Fatal.\n"); break; }


        // Get GLXFBConfig for Window and Pbuffer
        int fbcount;
        GLXFBConfig *configs;
        int visual_attribs[] = {
            GLX_X_RENDERABLE, True,
            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT | GLX_PBUFFER_BIT, // Need both types
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
            GLX_DEPTH_SIZE, 24, GLX_STENCIL_SIZE, 8, // Add depth/stencil for basic rendering capabilities
            GLX_DOUBLEBUFFER, True, // Double buffering needed for window rendering
            None
        };
        configs = local_glXChooseFBConfig(data.glx_dpy, screen, visual_attribs, &fbcount);
        if (fbcount == 0 || !configs) {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to get GLXFBConfig for Window/Pbuffer.\n");
            break; // Exit setup block
        }
        data.glx_fb_config = configs[0];
        XFree(configs);
        fprintf(stdout, "GLX Capture/Render Thread: Found %d FBConfigs, using first one for Window/Pbuffer.\n", fbcount);

        // Create GLX context (modern core profile preferred)
        static const int context_attribs[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3, // Try higher versions first (4.x is common)
            GLX_CONTEXT_MINOR_VERSION_ARB, 1,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        if (local_glXCreateContextAttribsARB) {
            data.glx_ctx = local_glXCreateContextAttribsARB(data.glx_dpy, data.glx_fb_config, NULL, True, context_attribs);
        }

        if (!data.glx_ctx) {
             fprintf(stdout, "GLX Capture/Render Thread: OpenGL 4.5 core context creation failed, trying 3.3 core.\n");
             static const int context_attribs_3_3[] = {
                 GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                 GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
                 None
             };
             if (local_glXCreateContextAttribsARB) {
                  data.glx_ctx = local_glXCreateContextAttribsARB(data.glx_dpy, data.glx_fb_config, NULL, True, context_attribs_3_3);
             }
        }

        if (!data.glx_ctx) {
             fprintf(stdout, "GLX Capture/Render Thread: OpenGL 3.3 core context creation failed, trying legacy context.\n");
             // Fallback to legacy context if modern creation fails
             XVisualInfo *vi = glXGetVisualFromFBConfig(data.glx_dpy, data.glx_fb_config);
             if (vi) {
                data.glx_ctx = glXCreateContext(data.glx_dpy, vi, NULL, True);
                XFree(vi);
             }
        }

        if (!data.glx_ctx) {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to create any GLX context.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread: GLX context created (%p).\n", data.glx_ctx);


        // Get VisualInfo from FBConfig to create X11 Window (needed by XCreateWindow)
        XVisualInfo *vi = glXGetVisualFromFBConfig(data.glx_dpy, data.glx_fb_config);
        if (!vi) {
             fprintf(stderr, "GLX Capture/Render Thread: Failed to get XVisualInfo from GLXFBConfig.\n");
             break; // Exit setup block
        }

        // Create X11 window for rendering
        XSetWindowAttributes swa;
        swa.colormap = XCreateColormap(data.glx_dpy, root_win, vi->visual, AllocNone);
        swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask | ClientMessage; // Add ClientMessage for window close
        swa.background_pixmap = None;
        swa.border_pixel = 0;

        data.glx_win = XCreateWindow(
            data.glx_dpy, root_win,
            0, 0, // position
            1280, 720, // initial size - make it smaller for easier debugging
            0, vi->depth, InputOutput, vi->visual,
            CWColormap | CWEventMask | CWBackPixmap | CWBorderPixel, &swa
        );

        XFree(vi); // Free visual info after window creation

        if (data.glx_win == 0) {
             fprintf(stderr, "GLX Capture/Render Thread: Failed to create X11 window.\n");
             break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread: X11 window created (%lu).\n", data.glx_win);

        // Set window title
        XStoreName(data.glx_dpy, data.glx_win, "NvFBC Capture Output");

        // Handle window close requests
        data.wm_delete_window = XInternAtom(data.glx_dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(data.glx_dpy, data.glx_win, &data.wm_delete_window, 1);

        // Map the window
        XMapWindow(data.glx_dpy, data.glx_win);
        XFlush(data.glx_dpy);


        // Create GLX Pbuffer (dummy surface to make context current for NvFBC calls)
        static const int pbuffer_attribs[] = { GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, None }; // Small but non-zero size
        if (local_glXCreatePbuffer) {
             data.glx_pbuffer = local_glXCreatePbuffer(data.glx_dpy, data.glx_fb_config, pbuffer_attribs);
        } else {
             // Fallback if extension not found, though this extension should be common
             fprintf(stderr, "GLX Capture/Render Thread: glXCreatePbuffer extension not found, attempting direct glXCreatePbuffer (unlikely to work for FBConfigs).\n");
             data.glx_pbuffer = glXCreatePbuffer(data.glx_dpy, data.glx_fb_config, pbuffer_attribs); // This might fail
        }

        if (!data.glx_pbuffer) {
          fprintf(stderr, "GLX Capture/Render Thread: Failed to create GLX Pbuffer surface. This might cause issues with NvFBC using an external context.\n");
          // Not necessarily fatal if NvFBC somehow works without Pbuffer or with window drawable, but bad practice.
          // Let's make it fatal for this example for clarity.
           break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread: GLX Pbuffer created (%lu).\n", data.glx_pbuffer);

        // Now that context is created, get the glXMakeContextCurrent pointer if available
        // We already tried once, but doing it again *after* context creation is more robust.
        // Update the global pointer used in the loop.
        p_glXMakeContextCurrent = (PFNGLXMAKECONTEXTCURRENTPROC)glXGetProcAddress((const GLubyte*)"glXMakeContextCurrent");
        if (!p_glXMakeContextCurrent) { fprintf(stdout, "GLX Capture/Render Thread: glXMakeContextCurrent still not found, will use glXMakeCurrent.\n"); }


        // Make GLX context current with the Pbuffer *for NvFBC handle creation and GL function loading*
        Bool make_current_ok = False;
        if (p_glXMakeContextCurrent) {
            make_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
        } else {
             make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
        }

        if (!make_current_ok) {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Pbuffer for handle creation/GL loading.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread: GLX context is current with Pbuffer.\n");
        // Use direct GL calls now that the context is current
        printf("GLX Context OpenGL Version: %s\n", glGetString(GL_VERSION));
        printf("GLX Context OpenGL Renderer: %s\n", glGetString(GL_RENDERER));


        // --- 2. Create NVFBC Handle ---
        // Get NvFBCCreateInstance function pointer from the shared global
        if (!s_NvFBCCreateInstance_ptr) {
             fprintf(stderr, "GLX Capture/Render Thread: NvFBCCreateInstance function pointer not available. Fatal.\n");
             // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
             break; // Exit setup block
        }

        pthread_mutex_lock(&data.session_state_mutex);
        data.nvFBC_api.dwVersion = NVFBC_VERSION; // Reset version just in case
        NVFBC_CREATE_HANDLE_PARAMS handle_params;
        memset(&handle_params, 0, sizeof(handle_params));
        handle_params.dwVersion = NVFBC_CREATE_HANDLE_PARAMS_VER;
        handle_params.bExternallyManagedContext = NVFBC_TRUE; // We manage contexts
        handle_params.glxCtx = data.glx_ctx; // Pass the GLX context we created
        handle_params.glxFBConfig = data.glx_fb_config;
        // NvFBC requires an 'enable key' for certain features, including ToGL on Quadro/RTX.
        // This key is usually provided with the NvFBC SDK documentation or support channel.
        // USING A GENERIC KEY FOUND ONLINE IS NOT GUARANTEED TO WORK AND IS NOT SUPPORTED.
        // Replace with your valid key if you have one. Using a placeholder here.
        // If you don't have a key, you might be limited to basic capture modes or Quadro/RTX might not work.
        const uint8_t enable_key[] = { 0xac, 0x10, 0xc9, 0x2e, 0xa5, 0xe6, 0x87, 0x4f, 0x8f, 0x4b, 0xf4, 0x61, 0xf8, 0x56, 0x27, 0xe9 };
        // !!! IMPORTANT: Replace the placeholder key above with your actual NvFBC enable key if required by your license/driver !!!
        // Using a zero-filled key might work for some basic cases or older drivers,
        // but typically Quadro/RTX with ToGL requires a valid key.
        handle_params.privateData = (void*)enable_key; // const_cast might be needed depending on struct definition
        handle_params.privateDataSize = sizeof(enable_key);

        // Call NvFBCCreateInstance using the pointer loaded in main
        NVFBCSTATUS ret = s_NvFBCCreateInstance_ptr(&data.nvFBC_api);

        if (ret != NVFBC_SUCCESS) {
            fprintf(stderr, "GLX Capture/Render Thread: NvFBCCreateInstance failed\n");
            pthread_mutex_unlock(&data.session_state_mutex);
             // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
            break; // Exit setup block
        }

         // Check if the essential NvFBC API functions are populated after instance creation
        if (!data.nvFBC_api.nvFBCCreateHandle || !data.nvFBC_api.nvFBCDestroyHandle ||
            !data.nvFBC_api.nvFBCCreateCaptureSession || !data.nvFBC_api.nvFBCDestroyCaptureSession ||
            !data.nvFBC_api.nvFBCToGLSetUp || !data.nvFBC_api.nvFBCToGLGrabFrame ||
            !data.nvFBC_api.nvFBCGetLastErrorStr)
        {
             fprintf(stderr, "GLX Capture/Render Thread: NvFBCCreateInstance succeeded, but some required NvFBC function pointers are NULL. API mismatch?\n");
             // Consider this a fatal setup error as NvFBC won't work
             pthread_mutex_unlock(&data.session_state_mutex);
              // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
             break; // Exit setup block
        }
        fprintf(stdout, "NvFBC API function pointers verified.\n");

        // Now create the actual NvFBC handle using the populated API struct
        // The context must be current for nvFBCCreateHandle when using bExternallyManagedContext = TRUE
         ret = data.nvFBC_api.nvFBCCreateHandle(&data.nvfbc_session, &handle_params);

        pthread_mutex_unlock(&data.session_state_mutex); // Unlock early if handle creation failed immediately


        if (ret != NVFBC_SUCCESS) {
             // Use the loaded function pointer for error string
            fprintf(stderr, "GLX Capture/Render Thread: nvFBCCreateHandle failed: %s\n", data.nvFBC_api.nvFBCGetLastErrorStr(NULL)); // Pass NULL as handle might be invalid
            // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread: NVFBC handle created (%lu).\n", data.nvfbc_session);


        // --- 3. Setup OpenGL Rendering Resources (Shaders, VAO/VBO) ---
        // Context should still be current with Pbuffer, which is fine for creating GL objects
        data.shader_program = createShaderProgram(vertexShaderSource, fragmentShaderSource);
        if (data.shader_program == 0) {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to create shader program. Fatal.\n");
             // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
            break; // Exit setup block
        }

        // Get the location of the sampler uniform *once*
        data.texture_sampler_location = glGetUniformLocation(data.shader_program, "screenTexture");
        if (data.texture_sampler_location == -1) {
             fprintf(stderr, "GLX Capture/Render Thread: Warning: Uniform 'screenTexture' not found in shader.\n");
             // Not fatal, but texture won't be sampled correctly without this
        }

        // Minimal VAO/VBO for a quad
        // --- CORRECTED TEXTURE COORDINATES ---
        // Assuming NvFBC provides texture data with the origin (0,0) at the TOP-LEFT,
        // we need to map the quad's vertices (bottom-left [-1,-1] to top-right [1,1])
        // to the texture coordinates such that the quad's top edge maps to the texture's
        // top edge (V=0) and the quad's bottom edge maps to the texture's bottom edge (V=1).
        float vertices[] = {
            // Positions (X, Y, Z) // Texture Coords (U, V)
            -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // Bottom-left of quad -> maps to (0, 1) in texture (bottom-left)
             1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // Bottom-right of quad -> maps to (1, 1) in texture (bottom-right)
             1.0f,  1.0f, 0.0f, 1.0f, 0.0f, // Top-right of quad    -> maps to (1, 0) in texture (top-right)
            -1.0f,  1.0f, 0.0f, 0.0f, 0.0f  // Top-left of quad     -> maps to (0, 0) in texture (top-left)
        };

        glGenVertexArrays(1, &data.quad_vao);
        glGenBuffers(1, &data.quad_vbo);

        glBindVertexArray(data.quad_vao);
        glBindBuffer(GL_ARRAY_BUFFER, data.quad_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // Texture coord attribute (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        fprintf(stdout, "OpenGL rendering resources created.\n");

        GLenum setup_err = glGetError();
        if (setup_err != GL_NO_ERROR) {
            fprintf(stderr, "GL error during OpenGL setup: %x\n", setup_err);
             // Unmake context before breaking
             if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
             else glXMakeCurrent(data.glx_dpy, None, NULL);
            break; // Exit setup block on GL error during setup
        }

        // Unmake GLX context current after setup (it will be made current again in the loop)
         if (p_glXMakeContextCurrent) {
             p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
         } else {
              glXMakeCurrent(data.glx_dpy, None, NULL);
         }
        fprintf(stdout, "GLX Capture/Render Thread: Unbound GLX context after setup.\n");


        // If we reach here, setup was successful
        setup_success = true;

    } while (false); // End of setup block

    if (!setup_success) {
         fprintf(stderr, "GLX Capture/Render Thread: Setup failed. Signaling stop.\n");
         data.stop_threads.store(true); // Signal main thread to stop on setup failure
    }


    // --- Main Capture and Render Loop ---
    // Only run the loop if setup was successful
    if (setup_success) {

        // Get initial window dimensions for viewport
        XWindowAttributes win_attrs;
        if (XGetWindowAttributes(data.glx_dpy, data.glx_win, &win_attrs)) {
             // Viewport is set when context is made current on the window later
            fprintf(stdout, "GLX Capture/Render Thread: Initial window size %dx%d.\n", win_attrs.width, win_attrs.height);
        } else {
            fprintf(stderr, "GLX Capture/Render Thread: Failed to get initial window attributes.\n");
        }

        // Calculate sleep duration based on FPS
        long sleep_us = (data.fps > 0) ? (1000000 / data.fps) : 1000; // Default to 1ms if FPS is 0 or negative

        while (!data.stop_threads.load()) {

            // --- Handle X Events ---
            XEvent event;
            // Process all pending events to avoid lag
            while (XPending(data.glx_dpy)) {
                XNextEvent(data.glx_dpy, &event);
                switch (event.type) {
                    case ClientMessage:
                         // Check for window close using the atom
                        if (event.xclient.message_type == XInternAtom(data.glx_dpy, "WM_PROTOCOLS", False) &&
                           (Atom)event.xclient.data.l[0] == data.wm_delete_window) {
                            fprintf(stdout, "GLX Capture/Render Thread: Window close requested.\n");
                            data.stop_threads.store(true);
                        }
                        break;
                    case ConfigureNotify:
                         // Handle window resize
                        if (event.xconfigure.width > 0 && event.xconfigure.height > 0) {
                            // Viewport will be updated when context is made current on window
                             //fprintf(stdout, "GLX Capture/Render Thread: Window resized to %dx%d.\n", event.xconfigure.width, event.xconfigure.height);
                        }
                        break;
                    case Expose:
                        // Window exposed, force redraw. A simple redraw is handled by the main loop's rendering path
                        break;
                    default:
                        break;
                }
            }
            if (data.stop_threads.load()) break; // Exit loop if stop requested by event handler

            // Check for recreate signal (simulating settings change)
            if (data.recreate_session.load()) {
                data.recreate_session.store(false);
                fprintf(stdout, "GLX Capture/Render Thread: Recreating NVFBC session...\n");

                // Destroy existing session if any
                pthread_mutex_lock(&data.session_state_mutex);
                if (data.has_capture_session.load()) {
                    NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                    // Context must be current with Pbuffer for this call
                    Bool make_pbuffer_current_ok = False;
                     if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) {
                          if (p_glXMakeContextCurrent) { make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx); }
                          else { make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); }
                     }
                     if (!make_pbuffer_current_ok) { fprintf(stderr, "GLX Capture/Render Thread: Failed to make context current for DestroySession.\n"); }


                    if (data.nvfbc_session != -1 && nvFBC->nvFBCDestroyCaptureSession) { // Ensure handle and function pointer are valid
                         NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params);
                         if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture/Render Thread: nvFBCDestroyCaptureSession failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                         else fprintf(stdout, "GLX Capture/Render Thread: Old NVFBC session destroyed.\n");
                    } else {
                         fprintf(stdout, "GLX Capture/Render Thread: NvFBC session handle invalid or DestroySession function not available, skipping DestroyCaptureSession during recreate.\n");
                    }

                    data.has_capture_session.store(false);
                    data.current_grabbed_texture = 0; // Mark texture as invalid

                     // Unmake context after call
                     if (data.glx_dpy && make_pbuffer_current_ok) { // Only unmake if we successfully made it current
                          if (p_glXMakeContextCurrent) { p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL); }
                          else { glXMakeCurrent(data.glx_dpy, None, NULL); }
                     }
                }
                pthread_mutex_unlock(&data.session_state_mutex);

                // Only attempt creation if the handle exists and the API functions are available
                if (data.nvfbc_session != -1 && nvFBC->nvFBCCreateCaptureSession && nvFBC->nvFBCToGLSetUp)
                {
                    // Create new session with hardcoded settings
                    NVFBC_CREATE_CAPTURE_SESSION_PARAMS cap_params = {
                        .dwVersion = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER,
                        .eCaptureType = NVFBC_CAPTURE_TO_GL,
                        .eTrackingType = data.screen_id == -1 ? NVFBC_TRACKING_SCREEN : NVFBC_TRACKING_OUTPUT,
                        .dwOutputId = (uint32_t)data.screen_id, // Cast needed
                        .bWithCursor = NVFBC_FALSE, // Simplicity
                        .bDisableAutoModesetRecovery = NVFBC_FALSE,
                        .bRoundFrameSize = NVFBC_FALSE,
                        // Calculate sampling rate in ms from desired FPS
                        .dwSamplingRateMs = (uint32_t)((data.fps > 0 ? (1000.0 / data.fps) : 33) + 0.5),
                        .bPushModel = NVFBC_TRUE, // Non-blocking grab
                        .bAllowDirectCapture = NVFBC_TRUE, // Try direct capture
                    };
                    // Clamp sampling rate to reasonable range if needed (e.g., 1ms to 1000ms)
                    if (cap_params.dwSamplingRateMs == 0) cap_params.dwSamplingRateMs = 1;


                    pthread_mutex_lock(&data.session_state_mutex);
                    NVFBCSTATUS ret = nvFBC->nvFBCCreateCaptureSession(data.nvfbc_session, &cap_params);

                    // Check for success before attempting setup
                    if (ret == NVFBC_SUCCESS) {
                        fprintf(stdout, "GLX Capture/Render Thread: NVFBC capture session created successfully.\n");

                        // Setup TO_GL
                        data.togl_setup_params = (NVFBC_TOGL_SETUP_PARAMS){
                            .dwVersion = NVFBC_TOGL_SETUP_PARAMS_VER,
                            .eBufferFormat = NVFBC_BUFFER_FORMAT_RGBA, //NVFBC_BUFFER_FORMAT_RGBA, // Standard format
                            .bWithDiffMap = NVFBC_FALSE, // Simplicity
                            .dwDiffMapScalingFactor = 1,
                            // dwTextures is output by nvFBCToGLSetUp
                        };
                        // GLX context MUST be current with Pbuffer for this call if bExternallyManagedContext is TRUE
                         Bool make_pbuffer_current_ok = False;
                         if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) {
                              if (p_glXMakeContextCurrent) { make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx); }
                              else { make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); }
                         }
                         if (!make_pbuffer_current_ok) {
                             fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Pbuffer for nvFBCToGLSetUp! Setup might fail.\n");
                             // Proceed, but setup will likely fail
                         }

                        NVFBCSTATUS setup_ret = nvFBC->nvFBCToGLSetUp(data.nvfbc_session, &data.togl_setup_params); // Use loaded pointer

                        // Unmake GLX context current after setup
                         if (data.glx_dpy && make_pbuffer_current_ok) { // Only unmake if we successfully made it current
                              if (p_glXMakeContextCurrent) { p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL); }
                              else { glXMakeCurrent(data.glx_dpy, None, NULL); }
                         }

                        if (setup_ret == NVFBC_SUCCESS) {
                            fprintf(stdout, "GLX Capture/Render Thread: nvFBCToGLSetUp succeeded.\n");
                            // Check if we actually got textures back
                            bool textures_ok = false;
                            for(int i=0; i<NVFBC_TOGL_MAX_BUFFERS; ++i) {
                                if (data.togl_setup_params.dwTextures[i] != 0) {
                                     textures_ok = true;
                                     break;
                                }
                            }
                            if (textures_ok) {
                                fprintf(stdout, "GLX Capture/Render Thread: Received valid texture IDs from NvFBC.\n");
                                data.has_capture_session.store(true); // Mark session as fully ready
                            } else {
                                fprintf(stderr, "GLX Capture/Render Thread: nvFBCToGLSetUp succeeded but returned no valid texture IDs.\n");
                                // Treat as setup failure
                                data.has_capture_session.store(false);
                                data.current_grabbed_texture = 0; // Ensure texture is invalid
                                // Attempt to destroy session if it was created successfully but setup failed
                                NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                                if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) {
                                    if (p_glXMakeContextCurrent) { make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx); }
                                    else { make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); }
                                } else { make_pbuffer_current_ok = False; } // Indicate context not made current
                                if (data.nvfbc_session != -1 && nvFBC->nvFBCDestroyCaptureSession) {
                                    NVFBCSTATUS dest_ret_after_setup = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params);
                                    if (dest_ret_after_setup != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture/Render Thread: nvFBCDestroyCaptureSession failed after setup failure: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                                    else fprintf(stdout, "GLX Capture/Render Thread: NVFBC session destroyed after setup failure.\n");
                                }
                                if (data.glx_dpy && make_pbuffer_current_ok) {
                                    if (p_glXMakeContextCurrent) { p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL); }
                                    else { glXMakeCurrent(data.glx_dpy, None, NULL); }
                                }
                            }

                        } else { // nvFBCToGLSetUp failed
                            fprintf(stderr, "GLX Capture/Render Thread: nvFBCToGLSetUp failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                            data.has_capture_session.store(false);
                            data.current_grabbed_texture = 0; // Ensure texture is invalid
                            // Attempt to destroy session if it was created successfully but setup failed
                            NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                            if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) {
                                if (p_glXMakeContextCurrent) { make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx); }
                                else { make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); }
                            } else { make_pbuffer_current_ok = False; } // Indicate context not made current
                            if (data.nvfbc_session != -1 && nvFBC->nvFBCDestroyCaptureSession) {
                                NVFBCSTATUS dest_ret_after_setup = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params);
                                if (dest_ret_after_setup != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture/Render Thread: nvFBCDestroyCaptureSession failed after setup failure: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                                else fprintf(stdout, "GLX Capture/Render Thread: NVFBC session destroyed after setup failure.\n");
                            }
                            if (data.glx_dpy && make_pbuffer_current_ok) {
                                if (p_glXMakeContextCurrent) { p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL); }
                                else { glXMakeCurrent(data.glx_dpy, None, NULL); }
                            }
                        }
                    } else { // nvFBCCreateCaptureSession failed
                        fprintf(stderr, "GLX Capture/Render Thread: nvFBCCreateCaptureSession failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                         data.has_capture_session.store(false);
                         data.current_grabbed_texture = 0; // Ensure texture is invalid
                    }

                    pthread_mutex_unlock(&data.session_state_mutex);

                    // If session setup failed, wait briefly before retrying
                    if (!data.has_capture_session.load()) {
                        usleep(100000); // Wait 100ms before retrying session creation
                    } else {
                        fprintf(stdout, "GLX Capture/Render Thread: NVFBC session recreation attempt complete.\n");
                    }

                } else {
                     // If handle or API functions are missing, we can't even try to recreate
                     fprintf(stderr, "GLX Capture/Render Thread: Cannot recreate session: NvFBC handle or API functions missing.\n");
                     usleep(100000); // Prevent busy-waiting
                }
                continue; // Skip to next loop iteration after recreate attempt
            } // end recreate_session check

            if (!data.has_capture_session.load() || data.nvfbc_session == -1 || nvFBC->nvFBCToGLGrabFrame == NULL) {
                // Wait if session creation failed or is pending, or API not ready
                usleep(10000);
                continue; // Skip to next loop iteration
            }

            // --- Grab Frame ---
            // Make GLX context current with the Pbuffer for NvFBC grab call
            Bool make_pbuffer_current_ok = False;
            if (p_glXMakeContextCurrent) {
                 make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
            } else {
                 make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
            }

            if (!make_pbuffer_current_ok) {
                fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Pbuffer for grab! Skipping grab.\n");
                // We can't grab without the context, but maybe we can still render the last frame?
                // No, context is needed for *any* GL call. This is fatal for the loop iteration.
                 // Attempt to unmake before continuing
                 if (data.glx_dpy) { // Need display to unmake
                      if (p_glXMakeContextCurrent) p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
                      else glXMakeCurrent(data.glx_dpy, None, NULL);
                 }
                 usleep(sleep_us / 2); // Wait a bit before retrying
                continue; // Skip to rendering the previous frame (or nothing if current_grabbed_texture is 0)
            }

            GLuint grabbed_gl_texture_id = 0; // Local var for the texture ID from the grab
            NVFBC_FRAME_GRAB_INFO info = {0};
            NVFBC_TOGL_GRAB_FRAME_PARAMS grab_params = {
                .dwVersion = NVFBC_TOGL_GRAB_FRAME_PARAMS_VER,
                .dwFlags = NVFBC_TOGL_GRAB_FLAGS_NOWAIT | NVFBC_TOGL_GRAB_FLAGS_FORCE_REFRESH, // Use NOWAIT for push model
                .pFrameGrabInfo = &info,
                .dwTimeoutMs = 0 // Timeout is ignored with NOWAIT
            };

            pthread_mutex_lock(&data.session_state_mutex); // Protect NvFBC session handle and togl_setup_params access during grab
            NVFBCSTATUS ret = nvFBC->nvFBCToGLGrabFrame(data.nvfbc_session, &grab_params); // Use loaded pointer
            // Texture ID access happens *within* the mutex block if a new frame was grabbed successfully

            bool frame_grabbed_new = false; // Flag for newly grabbed frame
            if (ret == NVFBC_SUCCESS && info.bIsNewFrame == NVFBC_TRUE) {
                grabbed_gl_texture_id = data.togl_setup_params.dwTextures[grab_params.dwTextureIndex];
                 // Check if the texture index is valid
                 // if (grab_params.dwTextureIndex < NVFBC_TOGL_MAX_BUFFERS) {
                 //    // Get the GL texture ID from the setup parameters struct
                 //    grabbed_gl_texture_id = data.togl_setup_params.dwTextures[grab_params.dwTextureIndex];
                 // } else {
                 //     fprintf(stderr, "GLX Capture/Render Thread: Grab succeeded but returned invalid texture index %u (max %u).\n", grab_params.dwTextureIndex, NVFBC_TOGL_MAX_BUFFERS - 1);
                 //     grabbed_gl_texture_id = 0; // Ensure grabbed_gl_texture_id is 0 if index is bad
                 // }

                 if (grabbed_gl_texture_id == 0) {
                      frame_grabbed_new = true;
                      // Update dimensions atomically
                      data.frame_width.store(info.dwWidth);
                      data.frame_height.store(info.dwHeight);
                      // Store the grabbed texture ID for rendering
                      data.current_grabbed_texture = grabbed_gl_texture_id; // Update shared state
                      //fprintf(stdout, "GLX Grabbed new frame, texture ID: %u, size: %ux%u\n", data.current_grabbed_texture, info.dwWidth, info.dwHeight);
                 } else {
                     // This happens if grab succeeds but no valid texture is available (e.g., index 0 invalid)
                     // Or if the index was out of bounds and we set grabbed_gl_texture_id to 0.
                     fprintf(stderr, "GLX Capture/Render Thread: Grab succeeded but returned texture ID 0 or invalid index.\n");
                 }
            } else if (ret == NVFBC_ERROR_GL_TIMEOUT) {
                 // Expected with NOWAIT if no new frame. Keep the last grabbed texture ID.
                 //fprintf(stdout, "GLX Capture/Render Thread: Grab timeout (no new frame).\n");
                 // No new frame, so grabbed_gl_texture_id remains 0. We will render the previous frame if available.
            } else if (ret == NVFBC_ERROR_GL_PRECONDITION_NOT_MET) {
                 // This can happen if context is not current or session is invalid.
                 fprintf(stderr, "GLX Capture/Render Thread: nvFBCToGLGrabFrame failed (PRECONDITION_NOT_MET): %s. Forcing session recreate.\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                 data.recreate_session.store(true); // Signal recreate
                 data.current_grabbed_texture = 0; // Invalidate texture
            }
            else if (ret != NVFBC_SUCCESS) {
                 // Other errors are unexpected
                 fprintf(stderr, "GLX Capture/Render Thread: nvFBCToGLGrabFrame failed: %s. Forcing session recreate.\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
                 // In case of severe error, reset texture and signal recreate
                 data.current_grabbed_texture = 0;
                 data.recreate_session.store(true); // Signal recreate
            }

            pthread_mutex_unlock(&data.session_state_mutex);


            // Unmake GLX context current after NvFBC grab
            if (data.glx_dpy && make_pbuffer_current_ok) { // Only unmake if we successfully made it current
                if (p_glXMakeContextCurrent) {
                    p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
                } else {
                     glXMakeCurrent(data.glx_dpy, None, NULL);
                }
            }


            // --- Render the grabbed frame (if available) ---
            // Make GLX context current with the WINDOW for rendering
             Bool make_window_current_ok = False;
             if (data.glx_dpy && data.glx_ctx && data.glx_win) { // Ensure drawables are valid
                 if (p_glXMakeContextCurrent) {
                     make_window_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_win, data.glx_win, data.glx_ctx);
                 } else {
                     make_window_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_win, data.glx_ctx);
                 }
             }

             if (!make_window_current_ok) {
                 fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Window for rendering! Skipping render this frame.\n");
                 // No context for rendering, just continue the loop after a short sleep
                 usleep(sleep_us / 2);
                 continue;
             }

            // Get current window dimensions for viewport
            XWindowAttributes current_win_attrs;
            if (XGetWindowAttributes(data.glx_dpy, data.glx_win, &current_win_attrs) && current_win_attrs.width > 0 && current_win_attrs.height > 0) {
                 glViewport(0, 0, current_win_attrs.width, current_win_attrs.height);
            } else {
                 // Fallback or use a default viewport if attributes can't be fetched
                 fprintf(stderr, "GLX Capture/Render Thread: Could not get current window attributes for viewport. Using default 1280x720.\n");
                 glViewport(0, 0, 1280, 720); // Default
            }

            // Perform rendering
            if (/*data.current_grabbed_texture != 0 && */data.shader_program != 0 && data.quad_vao != 0) {
                 glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Grayish background
                 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Also clear depth if using depth testing

                 glUseProgram(data.shader_program);

                 // Set the texture uniform to use texture unit 0
                 if (data.texture_sampler_location != -1) {
                     glUniform1i(data.texture_sampler_location, 0); // Use GL_TEXTURE0
                 } else {
                     fprintf(stderr, "GLX Capture/Render Thread: Cannot set texture uniform, location is -1.\n");
                 }

                 // Activate texture unit 0 and bind the grabbed texture
                 glActiveTexture(GL_TEXTURE0);
                 glBindTexture(GL_TEXTURE_2D, data.current_grabbed_texture);

                 // --- Set Texture Parameters ---
                 // This is crucial for the texture to be sampleable. Do this *after* binding.
                 // NvFBC provides the texture ID, but doesn't necessarily set these in your context.
                 // Use direct GL calls for texture parameters
                 // ADDED: These lines are critical for the texture to be usable.
                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                 // Optional: Disable mipmaps if only level 0 is valid (common for capture)
                 // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);


                 // Draw the quad
                 glBindVertexArray(data.quad_vao);
                 glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // Use TRIANGLE_FAN or TRIANGLES (with indices)
                 glBindVertexArray(0);

                 // Clean up bindings
                 glBindTexture(GL_TEXTURE_2D, 0);
                 glUseProgram(0);

                 fprintf(stdout, "LOL....\n");
                 // Swap buffers to show the rendered frame (core GLX function)
                 glXSwapBuffers(data.glx_dpy, data.glx_win);

                 GLenum gl_err = glGetError();
                 if (gl_err != GL_NO_ERROR) {
                     fprintf(stderr, "GL error during rendering frame %u: %x\n", data.current_grabbed_texture, gl_err);
                 } // else { fprintf(stdout, "GLX Rendered frame %u\n", data.current_grabbed_texture); }

            } else {
                // If no valid texture or setup failed, just clear the window
                 glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
                 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Also clear depth
                 glXSwapBuffers(data.glx_dpy, data.glx_win); // Core GLX function
                 fprintf(stdout, "GLX Rendering blank frame.\n");
            }

            // Unmake GLX context current after rendering
            if (data.glx_dpy && make_window_current_ok) { // Only unmake if we successfully made it current
                if (p_glXMakeContextCurrent) {
                    p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
                } else {
                     glXMakeCurrent(data.glx_dpy, None, NULL);
                }
            }


            // Sleep for FPS control
            // Note: NOWAIT grabs might return the same frame multiple times.
            // Sleeping based on target FPS is a simple way to limit render rate.
            usleep(sleep_us);


        } // End of capture/render loop
    }


    // --- Cleanup ---
    fprintf(stdout, "GLX Capture/Render Thread: Cleaning up.\n");

    // Signal main thread to stop if we are exiting unexpectedly (e.g., setup failure)
    data.stop_threads.store(true);

    // Ensure NvFBC API functions are loaded before attempting to use them for cleanup
    if (nvFBC && nvFBC->nvFBCDestroyCaptureSession && nvFBC->nvFBCDestroyHandle && nvFBC->nvFBCGetLastErrorStr) {
        // Destroy NVFBC session and handle
        pthread_mutex_lock(&data.session_state_mutex);
        if (data.has_capture_session.load()) {
            NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
            // Context must be current with Pbuffer for this call
            Bool make_pbuffer_current_ok = False;
             if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) { // Check validity
                  if (p_glXMakeContextCurrent) {
                       make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
                  } else {
                       make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
                  }
                  if (!make_pbuffer_current_ok) {
                       fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Pbuffer for nvFBCDestroyCaptureSession! Cleanup might fail.\n");
                  }
             } else {
                 fprintf(stderr, "GLX Capture/Render Thread: Missing display, context or pbuffer for NvFBC session cleanup.\n");
             }

            if (data.nvfbc_session != -1) { // Ensure session handle is seemingly valid before calling
                 NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use loaded pointer
                 if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture/Render Thread: nvFBCDestroyCaptureSession failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session));
            } else {
                 fprintf(stdout, "GLX Capture/Render Thread: NvFBC session handle already invalid, skipping DestroyCaptureSession.\n");
            }
            data.has_capture_session.store(false);
            data.current_grabbed_texture = 0; // Texture is now invalid

             // Unmake context after call
             if (data.glx_dpy && make_pbuffer_current_ok) { // Only unmake if we successfully made it current
                  if (p_glXMakeContextCurrent) {
                      p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
                  } else {
                       glXMakeCurrent(data.glx_dpy, None, NULL);
                  }
             }
        }


        // Context might need to be current for DestroyHandle if bExternallyManagedContext was TRUE
        // Making GLX context current before destroying the handle (using the Pbuffer drawable is sufficient)
         Bool make_pbuffer_current_ok = False;
         if (data.glx_dpy && data.glx_ctx && data.glx_pbuffer) { // Check validity
              if (p_glXMakeContextCurrent) {
                   make_pbuffer_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
              } else {
                   make_pbuffer_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
              }
              if (!make_pbuffer_current_ok) {
                   fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current with Pbuffer for nvFBCDestroyHandle! Cleanup might fail.\n");
              }
         } else {
              fprintf(stderr, "GLX Capture/Render Thread: Missing display, context or pbuffer for NvFBC handle cleanup.\n");
         }


        if (data.nvfbc_session != -1) {
            NVFBC_DESTROY_HANDLE_PARAMS destroy_params = {.dwVersion = NVFBC_DESTROY_HANDLE_PARAMS_VER};
            NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyHandle(data.nvfbc_session, &destroy_params); // Use loaded pointer
            if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture/Render Thread: nvFBCDestroyHandle failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(NULL)); // Use NULL if handle might be bad
            data.nvfbc_session = -1;
        } else {
             fprintf(stdout, "GLX Capture/Render Thread: NvFBC handle already invalid, skipping DestroyHandle.\n");
        }

        // Unmake GLX context current after destroying handle
         if (data.glx_dpy && make_pbuffer_current_ok) { // Only unmake if we successfully made it current
              if (p_glXMakeContextCurrent) {
                  p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
              } else {
                   glXMakeCurrent(data.glx_dpy, None, NULL);
              }
         }

        pthread_mutex_unlock(&data.session_state_mutex);
    } else {
         fprintf(stderr, "GLX Capture/Render Thread: NvFBC API functions not fully available for cleanup.\n");
    }

    // Destroy OpenGL resources (Shaders, VAO/VBO) - requires context to be current (using window or pbuffer is fine)
    // Ensure GL function pointers are loaded before attempting GL cleanup calls
     // The functions like glDeleteProgram should be available directly from gl.h/glcorearb.h now
     if (data.glx_dpy && data.glx_ctx) { // Check display and context
          Bool make_current_ok = False;
          // Try making current on the window first, fallback to pbuffer if window is bad
          if (data.glx_win) {
               if (p_glXMakeContextCurrent) make_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_win, data.glx_win, data.glx_ctx);
               else make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_win, data.glx_ctx);
          }
          if (!make_current_ok && data.glx_pbuffer) {
               if (p_glXMakeContextCurrent) make_current_ok = p_glXMakeContextCurrent(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
               else make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
          }

           if (make_current_ok) {
               if (data.shader_program != 0) {
                   glDeleteProgram(data.shader_program);
                   data.shader_program = 0;
                   fprintf(stdout, "Shader program destroyed.\n");
               }
               if (data.quad_vao != 0) {
                    glDeleteVertexArrays(1, &data.quad_vao);
                    data.quad_vao = 0;
                    fprintf(stdout, "Quad VAO destroyed.\n");
               }
               if (data.quad_vbo != 0) {
                    glDeleteBuffers(1, &data.quad_vbo);
                    data.quad_vbo = 0;
                     fprintf(stdout, "Quad VBO destroyed.\n");
               }
               // The texture (data.current_grabbed_texture) is owned by NvFBC, we don't delete it here.
               // It is released when the NvFBC session is destroyed.

              // Unmake GLX context current after cleanup
               if (data.glx_dpy) {
                   if (p_glXMakeContextCurrent) {
                       p_glXMakeContextCurrent(data.glx_dpy, None, None, NULL);
                   } else {
                        glXMakeCurrent(data.glx_dpy, None, NULL);
                   }
               }
           } else {
               fprintf(stderr, "GLX Capture/Render Thread: Failed to make GLX context current for OpenGL resource cleanup. Resources might leak.\n");
           }
     } else {
         fprintf(stderr, "GLX Capture/Render Thread: Missing display or context for OpenGL resource cleanup.\n");
     }


    // Destroy GLX context and Drawables
    if (data.glx_ctx && data.glx_dpy) {
        glXDestroyContext(data.glx_dpy, data.glx_ctx);
        data.glx_ctx = NULL;
         fprintf(stdout, "GLX context destroyed.\n");
    }
    if (data.glx_pbuffer && data.glx_dpy) {
        glXDestroyPbuffer(data.glx_dpy, data.glx_pbuffer);
        data.glx_pbuffer = None;
         fprintf(stdout, "GLX Pbuffer destroyed.\n");
    }
    if (data.glx_win && data.glx_dpy) {
        XDestroyWindow(data.glx_dpy, data.glx_win);
        data.glx_win = 0;
        fprintf(stdout, "GLX window destroyed.\n");
    }


    // Close X11 display connection
    if (data.glx_dpy) {
        XCloseDisplay(data.glx_dpy);
        data.glx_dpy = NULL;
         fprintf(stdout, "GLX thread X11 display closed.\n");
    }


    fprintf(stdout, "GLX Capture/Render Thread: Exited.\n");
    return NULL;
}


// --- Main Application Functions ---



int main() {
    fprintf(stdout, "Minimal NvFBC GLX Capture/Render Test (Linux)\n");

    // --- 0. Initialize shared data mutex ---
    pthread_mutex_init(&data.session_state_mutex, NULL);

    bool main_setup_success = false;

    // Use separate flags for each resource created during setup
    pthread_t glx_thread_handle = 0;


    do { // Use a do-while(false) loop to allow using break for early exit during setup

        // --- Load NvFBC Library Dynamically ---
        fprintf(stdout, "Loading NvFBC library...\n");
        // Common library names: libnvidia-fbc.so.1, libnvidia-fbc.so
        const char* nvfbc_lib_names[] = {"libnvidia-fbc.so.1", "libnvidia-fbc.so"};
        for (size_t i = 0; i < sizeof(nvfbc_lib_names)/sizeof(nvfbc_lib_names[0]); ++i) {
             data.nvfbc_lib_handle = os_dlopen(nvfbc_lib_names[i]);
             if (data.nvfbc_lib_handle) {
                  fprintf(stdout, "Successfully loaded %s\n", nvfbc_lib_names[i]);
                  break;
             }
        }

        if (!data.nvfbc_lib_handle) {
            fprintf(stderr, "Error: Failed to load any NvFBC library (tried libnvidia-fbc.so.1, libnvidia-fbc.so). Please ensure the NVIDIA driver is installed and the library is in the linker path.\n");
            break; // Exit setup block
        }


        // --- Get NvFBCCreateInstance function pointer (needed by the GLX thread) ---
        s_NvFBCCreateInstance_ptr = (PFNNVFBCCreateInstance)os_dlsym(data.nvfbc_lib_handle, "NvFBCCreateInstance");

        if (!s_NvFBCCreateInstance_ptr) {
            fprintf(stderr, "Error: Failed to get NvFBCCreateInstance symbol from the loaded library. Symbol not found or mismatch.\n");
             // Attempt to close the library handle
            if (data.nvfbc_lib_handle) {
                 fprintf(stdout, "Main thread closing NvFBC library handle after symbol lookup failure.\n");
                 os_dlclose(data.nvfbc_lib_handle);
                 data.nvfbc_lib_handle = NULL;
            }
            break; // Exit setup block
        }
        fprintf(stdout, "NvFBCCreateInstance symbol found.\n");


        // --- 1. Create Capture/Render Thread ---
        fprintf(stdout, "Creating GLX Capture/Render thread...\n");

        int ret_glx = pthread_create(&glx_thread_handle, NULL, glx_capture_render_thread, NULL);
        if (ret_glx != 0) {
            fprintf(stderr, "Error: Failed to create GLX capture/render thread: %s\n", strerror(ret_glx));
            // Attempt to close the library handle if the thread failed to create
            if (data.nvfbc_lib_handle) {
                 fprintf(stdout, "Main thread closing NvFBC library handle after thread creation failure.\n");
                 os_dlclose(data.nvfbc_lib_handle);
                 data.nvfbc_lib_handle = NULL;
            }
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture/Render Thread created.\n");

        // If we reached here, main setup (loading lib, getting symbol, creating thread) was successful
        main_setup_success = true;

    } while (false); // End of setup block


    // --- 2. Main Loop (Wait for thread) ---
    if (main_setup_success) {
        fprintf(stdout, "Main thread waiting for GLX thread to finish (close the GLX window).\n");
        // The main thread simply waits for the capture/render thread to exit.
        // The capture/render thread handles its own window events to trigger stop.
        pthread_join(glx_thread_handle, NULL);
        fprintf(stdout, "GLX Capture/Render thread finished.\n");
    } else {
         fprintf(stderr, "Main thread setup failed. Exiting.\n");
    }


    // --- 3. Cleanup ---
    // NvFBC library handle is closed by the main thread if it was successfully loaded.
    // This happens outside the `main_setup_success` block so it cleans up even on setup failure.
    if (data.nvfbc_lib_handle) {
        fprintf(stdout, "Main thread closing NvFBC library handle during main cleanup.\n");
        os_dlclose(data.nvfbc_lib_handle);
        data.nvfbc_lib_handle = NULL;
    }

    // Destroy mutex (initialized unconditionally)
    pthread_mutex_destroy(&data.session_state_mutex);

    fprintf(stdout, "Cleanup complete. Exiting.\n");
    return 0;
}