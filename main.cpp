#define GL_GLEXT_PROTOTYPES // Required for GL extension prototypes (e.g. glEGLImageTargetTexture2DOES)
#define EGL_EGLEXT_PROTOTYPES // Required for EGL extension prototypes (e.g. eglCreateImageKHR)
#define GLX_GLXEXT_PROTOTYPES // Required for GLX extension prototypes (e.g. glXCreateContextAttribsARB)

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <GL/gl.h> // For basic GL calls like glBindTexture, glGetError, glGenTextures, glDeleteTextures
#include <GL/glext.h> // For GL_TEXTURE_2D and other constants

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h> // For sleep

#include <pthread.h>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
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
    // A more robust check might be needed in practice if NULL symbols are possible/expected.
    // For function pointers, NULL is usually an error.
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
         fprintf(stderr, "dlsym(%s) failed: %s\n", name, dlsym_error);
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

// Define function pointer type for NvFBCCreateInstance
typedef NVFBCSTATUS (*PFNNVFBCCreateInstance)(NVFBC_API_FUNCTION_LIST *pFunctions);

// Pointers to EGL extension functions - will be loaded via eglGetProcAddress in main thread
static PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR = NULL;
static PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR = NULL;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES = NULL;

// --- Thread-safe Queue ---
template <typename T>
class SharedQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond_var;
    std::atomic<bool> stop = false;

public:
    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(value);
        cond_var.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [this]{ return !queue.empty() || stop.load(); });
        if (stop.load() && queue.empty()) {
             // Return a sentinel value indicating stop. For EGLImageKHR, EGL_NO_IMAGE_KHR works.
             return EGL_NO_IMAGE_KHR;
        }
        if (queue.empty()) { // Should not happen based on wait condition, but defensive
             return EGL_NO_IMAGE_KHR;
        }
        T value = queue.front();
        queue.pop();
        return value;
    }

    void stop_queue() {
        stop.store(true);
        cond_var.notify_all();
    }

    size_t size() const {
         std::unique_lock<std::mutex> lock(mutex);
         return queue.size();
    }

    // Clear queue and destroy images (assuming T is EGLImageKHR)
    void clear(EGLDisplay dpy) {
        std::unique_lock<std::mutex> lock(mutex);
        while (!queue.empty()) {
            EGLImageKHR img = queue.front();
            queue.pop();
            // Only destroy if dpy is valid, function pointer is loaded, and image is not sentinel
            if (img != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY && p_eglDestroyImageKHR != NULL) {
                 if (p_eglDestroyImageKHR(dpy, img) != EGL_TRUE) {
                     fprintf(stderr, "Error destroying queued EGLImage %p during clear: %d\n", img, eglGetError());
                 }
            }
        }
    }
};


// --- Shared Data Structure ---
struct shared_data {
    std::atomic<bool> stop_threads = false;
    std::atomic<bool> recreate_session = true; // Simulate initial setup/update

    // NvFBC library handle and function list
    void *nvfbc_lib_handle = NULL;
    NVFBC_API_FUNCTION_LIST nvFBC_api = {.dwVersion = NVFBC_VERSION}; // This will hold pointers from dlsym

    // NVFBC Session (Accessed by Capture Thread, protected by mutex for create/destroy)
    NVFBC_SESSION_HANDLE nvfbc_session = -1;
    NVFBC_TOGL_SETUP_PARAMS togl_setup_params; // Protected by session_state_mutex
    std::atomic<bool> has_capture_session = false; // Protected by session_state_mutex for modification
    pthread_mutex_t session_state_mutex; // Protects nvfbc_session, has_capture_session, togl_setup_params

    // Capture Thread (GLX) handles
    Display *glx_dpy = NULL; // GLX thread's X11 display
    GLXContext glx_ctx = NULL; // GLX context for NvFBC operations
    GLXPbuffer glx_pbuffer = None; // Dummy Pbuffer for making GLX context current
    EGLContext glx_thread_egl_ctx = EGL_NO_CONTEXT; // EGL context for GLX thread (shared with main)

    // EGL Data (shared display, used by Processor and Main and GLX thread EGL ops)
    EGLDisplay shared_egl_dpy = EGL_NO_DISPLAY; // The single EGL display handle
    EGLConfig shared_egl_config = NULL; // Config chosen by main, used by GLX thread for sharing
    std::atomic<bool> main_egl_setup_complete = false; // Signal main EGL is ready

    // EGLImage Shared Queue (Capture produces, Processor consumes)
    SharedQueue<EGLImageKHR> imageQueue;

    // Shared state between EGL Processor Thread and Main Render Thread
    EGLImageKHR current_render_image = EGL_NO_IMAGE_KHR; // Protected by frame_state_mutex
    std::atomic<bool> new_image_available = false; // Signaled by EGL Processor
    std::atomic<uint32_t> frame_width = 0; // Updated atomically by GLX thread
    std::atomic<uint32_t> frame_height = 0; // Updated atomically by GLX thread
    pthread_mutex_t frame_state_mutex; // Protects current_render_image

    // Main Thread (EGL) handles - for minimal rendering
    Display *main_x_dpy = NULL; // The X11 display used by the main thread
    Window main_x_win = 0;      // The X11 window used by the main thread
    EGLContext main_egl_ctx = EGL_NO_CONTEXT; // The main EGL context (share_context for GLX thread)
    GLuint main_render_tex = 0; // Texture in main thread's GL context
    EGLSurface main_egl_surface = EGL_NO_SURFACE; // EGL surface for the window

    // Settings (minimal, hardcoded or read once)
    int screen_id = -1; // -1 for entire desktop
    int fps = 30;

    bool nvfbc_lib_loaded = false;
};

static struct shared_data data;


// --- GLX Capture Thread Function ---
static void *glx_capture_thread(void *param)
{
    fprintf(stdout, "GLX Capture Thread: Started.\n");



    bool setup_success = false; // Flag to track overall setup success

    PFNGLXMAKECONTEXTCURRENTPROC glXMakeContextCurrent_ptr = NULL;
    PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB_ptr = NULL;
    PFNGLXCREATEPBUFFERPROC glXCreatePbuffer_ptr = NULL;
    PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig_ptr = NULL;

    Bool make_current_ok = False;
    EGLSurface egl_pbuffer = EGL_NO_SURFACE;
    NVFBC_API_FUNCTION_LIST* nvFBC = NULL;

    do { // Use a do-while(false) loop to allow using break for early exit during setup

        data.nvfbc_lib_loaded = false;
        bool nvfbc_instance_created = false;

        fprintf(stdout, "Loading NvFBC library...\n");
        data.nvfbc_lib_handle = os_dlopen("libnvidia-fbc.so.1");
        if (!data.nvfbc_lib_handle) {
            fprintf(stderr, "Error: Failed to load libnvidia-fbc.so.1. Please ensure the NVIDIA driver is installed and the library is in the linker path.\n");
            break; // Exit setup block
        }
        data.nvfbc_lib_loaded = true;
        fprintf(stdout, "NvFBC library loaded.\n");

        // --- Get NvFBCCreateInstance function pointer ---
        PFNNVFBCCreateInstance NvFBCCreateInstance_ptr = (PFNNVFBCCreateInstance)os_dlsym(data.nvfbc_lib_handle, "NvFBCCreateInstance");

        if (!NvFBCCreateInstance_ptr) {
            fprintf(stderr, "Error: Failed to get NvFBCCreateInstance symbol from the loaded library. Symbol not found or mismatch.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "NvFBCCreateInstance symbol found.\n");

        // --- Call NvFBCCreateInstance via pointer to populate the struct ---
        data.nvFBC_api.dwVersion = NVFBC_VERSION; // Ensure version is set before calling
        NVFBCSTATUS ret = NvFBCCreateInstance_ptr(&data.nvFBC_api); // Call the loaded function

        if (ret != NVFBC_SUCCESS) {
            fprintf(stderr, "Error: NvFBCCreateInstance failed (via dlsym). Error code %d.\n", ret);
            break; // Exit setup block
        }
        nvfbc_instance_created = true;
        fprintf(stdout, "NvFBC instance created (via dlsym) and API functions populated.\n");

        // Check if the essential pointers are non-NULL after instance creation
        if (!data.nvFBC_api.nvFBCCreateHandle || !data.nvFBC_api.nvFBCDestroyHandle ||
            !data.nvFBC_api.nvFBCCreateCaptureSession || !data.nvFBC_api.nvFBCDestroyCaptureSession ||
            !data.nvFBC_api.nvFBCToGLSetUp || !data.nvFBC_api.nvFBCToGLGrabFrame ||
            !data.nvFBC_api.nvFBCGetLastErrorStr)
        {
             fprintf(stderr, "Error: NvFBCCreateInstance succeeded, but some required NvFBC function pointers are NULL. API mismatch?\n");
             // Consider this a fatal setup error as NvFBC won't work
             break; // Exit setup block
        }
        fprintf(stdout, "NvFBC API function pointers verified.\n");

        

        // Get pointer to the shared NVFBC API function list
        nvFBC = &data.nvFBC_api;




        // --- 1. Open X11 Display and create GLX context/Pbuffer ---
        data.glx_dpy = XOpenDisplay(NULL); // Direct call
        if (!data.glx_dpy) {
            fprintf(stderr, "GLX Capture Thread: Failed to open X11 display.\n");
            break; // Exit setup block
        }
        int screen = DefaultScreen(data.glx_dpy); // Direct call

        // Get GLX extension function pointers needed for setup
        glXMakeContextCurrent_ptr = (PFNGLXMAKECONTEXTCURRENTPROC)glXGetProcAddress((const GLubyte*)"glXMakeContextCurrent");
        glXCreateContextAttribsARB_ptr = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
        glXCreatePbuffer_ptr = (PFNGLXCREATEPBUFFERPROC)glXGetProcAddress((const GLubyte*)"glXCreatePbuffer");
        glXChooseFBConfig_ptr = (PFNGLXCHOOSEFBCONFIGPROC)glXGetProcAddress((const GLubyte*)"glXChooseFBConfig");

        if (!glXMakeContextCurrent_ptr) { fprintf(stdout, "GLX Capture Thread: glXMakeContextCurrent not found, will use glXMakeCurrent.\n"); }
        if (!glXCreateContextAttribsARB_ptr) { fprintf(stdout, "GLX Capture Thread: glXCreateContextAttribsARB not found, will try legacy context.\n"); }
        if (!glXCreatePbuffer_ptr) { fprintf(stdout, "GLX Capture Thread: glXCreatePbuffer not found, will try direct call.\n"); }
        if (!glXChooseFBConfig_ptr) { fprintf(stderr, "GLX Capture Thread: glXChooseFBConfig not found. Cannot select FBConfig. Fatal.\n"); break; }


        // Get GLXFBConfig for Pbuffer
        int fbcount;
        GLXFBConfig *configs;
        int visual_attribs[] = {
            GLX_X_RENDERABLE, True,
            GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
            GLX_DOUBLEBUFFER, True, // Needed for some older drivers/setups, even on Pbuffer
            None
        };
        configs = glXChooseFBConfig_ptr(data.glx_dpy, screen, visual_attribs, &fbcount);
        if (fbcount == 0 || !configs) {
            fprintf(stderr, "GLX Capture Thread: Failed to get GLXFBConfig for Pbuffer.\n");
            break; // Exit setup block
        }
        GLXFBConfig chosen_fb_config = configs[0];
        XFree(configs);
        fprintf(stdout, "GLX Capture Thread: Found %d FBConfigs, using first one for Pbuffer.\n", fbcount);

        // Create GLX context (modern core profile preferred)
        static const int context_attribs[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3, // Try higher versions if available
            GLX_CONTEXT_MINOR_VERSION_ARB, 1, // 4.5 is common
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        if (glXCreateContextAttribsARB_ptr) {
            data.glx_ctx = glXCreateContextAttribsARB_ptr(data.glx_dpy, chosen_fb_config, NULL, True, context_attribs);
        }

        if (!data.glx_ctx) {
             fprintf(stdout, "GLX Capture Thread: Modern context creation failed, trying OpenGL 3.1 core.\n");
             static const int context_attribs_3_1[] = {
                 GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 1,
                 GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
                 None
             };
             if (glXCreateContextAttribsARB_ptr) {
                  data.glx_ctx = glXCreateContextAttribsARB_ptr(data.glx_dpy, chosen_fb_config, NULL, True, context_attribs_3_1);
             }
        }

        if (!data.glx_ctx) {
             fprintf(stdout, "GLX Capture Thread: OpenGL 3.1 core context creation failed, trying legacy context.\n");
             // Fallback to legacy context if modern creation fails
             XVisualInfo *vi = glXGetVisualFromFBConfig(data.glx_dpy, chosen_fb_config); // Direct call - requires chosen_fb_config to be available
             if (vi) {
                data.glx_ctx = glXCreateContext(data.glx_dpy, vi, NULL, True); // Direct call
                XFree(vi); // Direct call
             }
        }

        if (!data.glx_ctx) {
            fprintf(stderr, "GLX Capture Thread: Failed to create any GLX context.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: GLX context created.\n");

        // Create GLX Pbuffer (dummy surface to make context current for NvFBC)
        static const int pbuffer_attribs[] = { GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, None }; // Small but non-zero size
        if (glXCreatePbuffer_ptr) {
             data.glx_pbuffer = glXCreatePbuffer_ptr(data.glx_dpy, chosen_fb_config, pbuffer_attribs);
        } else {
             data.glx_pbuffer = glXCreatePbuffer(data.glx_dpy, chosen_fb_config, pbuffer_attribs); // Direct call
        }

        if (!data.glx_pbuffer) {
          fprintf(stderr, "GLX Capture Thread: Failed to create GLX Pbuffer surface.\n");
          break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: GLX Pbuffer created.\n");

        // Make GLX context current with the Pbuffer
        make_current_ok = False;
        if (glXMakeContextCurrent_ptr) {
            make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
        } else {
             make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); // Direct call
        }

        if (!make_current_ok) {
            fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current with Pbuffer.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: GLX context is current.\n");
        printf("GLX Context OpenGL Version: %s\n",glGetString(GL_VERSION)); // Direct call
        printf("GLX Context OpenGL Renderer: %s\n",glGetString(GL_RENDERER)); // Direct call


        #if 1
        // --- 2. Create Shared EGL Context in this Thread ---
        // This context will share resources with the main thread's EGL context.
        // It must be created on the *same* EGLDisplay as the main thread's context.
        // Wait for main thread to signal its EGL setup is complete
        fprintf(stdout, "GLX Capture Thread: Waiting for main thread EGL setup...\n");
        while (!data.main_egl_setup_complete.load() && !data.stop_threads.load()) {
             usleep(10000); // Wait 10ms
        }
        if (data.stop_threads.load()) break; // Exit if stop requested while waiting

        // Now that main is ready, data.shared_egl_dpy, data.shared_egl_config, data.main_egl_ctx are valid
        if (data.shared_egl_dpy == EGL_NO_DISPLAY || data.shared_egl_config == NULL || data.main_egl_ctx == EGL_NO_CONTEXT) {
             fprintf(stderr, "GLX Capture Thread: Main thread EGL setup incomplete or failed (shared_egl_dpy=%p, shared_egl_config=%p, main_egl_ctx=%p). Cannot create shared EGL context.\n",
                     data.shared_egl_dpy, data.shared_egl_config, data.main_egl_ctx);
             break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: Main thread EGL setup complete. Creating shared EGL context.\n");

        // Create EGL context in this thread, sharing with the main thread's context
        // Use the EGLDisplay and EGLConfig from the main thread.
        // Try creating shared context with NULL attributes first for maximum compatibility
        data.glx_thread_egl_ctx = eglCreateContext(
            data.shared_egl_dpy,          // Use the SHARED EGL display
            data.shared_egl_config,       // Use the config from the main thread (valid on shared_egl_dpy)
            data.main_egl_ctx,            // Share with the main thread's context (created on shared_egl_dpy)
            NULL // Try NULL attributes for default context compatible with config/share_context
        );

        if (data.glx_thread_egl_ctx == EGL_NO_CONTEXT) {
             EGLint egl_err = eglGetError(); // Direct call
             fprintf(stderr, "GLX Capture Thread: Failed to create shared EGL context (on shared_egl_dpy) with NULL attributes: EGL error %x\n", egl_err);

             // If NULL attributes fail, try specifying version explicitly (e.g., GLES2, GL 3.3)
             // Use the same version attributes attempted in main, as share contexts often require matching versions
             static const EGLint context_attribs_gles2[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE }; // Try GLES 2
              data.glx_thread_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, data.main_egl_ctx, context_attribs_gles2);

              if (data.glx_thread_egl_ctx == EGL_NO_CONTEXT) {
                   fprintf(stdout, "GLX Capture Thread: Failed to create shared EGL context (GLES2), trying GL 3.3.\n");
                   static const EGLint context_attribs_gl33[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
                   data.glx_thread_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, data.main_egl_ctx, context_attribs_gl33);
              }
        }

        if (data.glx_thread_egl_ctx == EGL_NO_CONTEXT) {
             EGLint egl_err = eglGetError();
             fprintf(stderr, "GLX Capture Thread: Failed to create shared EGL context after trying multiple attributes: EGL error %x. Fatal.\n", egl_err);
             break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: Shared EGL context created (%p).\n", data.glx_thread_egl_ctx);


        EGLint pbufferAttribs[] = {
                EGL_WIDTH, 64,  // Specify desired width
                EGL_HEIGHT, 64, // Specify desired height
                EGL_NONE
            };
        
        egl_pbuffer = eglCreatePbufferSurface(data.shared_egl_dpy, data.shared_egl_config, pbufferAttribs);
        if (egl_pbuffer == EGL_NO_SURFACE) {
            fprintf(stderr, "Error: eglCreatePbufferSurface failed.");
            break;
        }
        fprintf(stdout, "Dummy pbuffer surface created.");

        // Unmake GLX context current after setup (it will be made current again in the loop)
         if (glXMakeContextCurrent_ptr) {
             glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
         } else {
              glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
         }
        fprintf(stdout, "GLX Capture Thread: Unbound GLX context after setup.\n");

        
        // --- 3. Create NVFBC Handle ---
        // Check if NvFBC API functions are loaded
        if (!nvFBC || !nvFBC->nvFBCCreateHandle) {
             fprintf(stderr, "GLX Capture Thread: NvFBC API functions not loaded correctly or missing nvFBCCreateHandle. Stopping.\n");
             break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: NvFBC API function list seems valid.\n");

        
        // Create NvFBC handle
        // The GLX context MUST be current when calling nvFBCCreateHandle if bExternallyManagedContext is TRUE
        if (glXMakeContextCurrent_ptr) {
         make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
        } else {
          make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); // Direct call
        }
        if (!make_current_ok) {
         fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current for nvFBCCreateHandle. Fatal.\n");
         break; // Exit setup block
        }
        #endif

        pthread_mutex_lock(&data.session_state_mutex);
        NVFBC_CREATE_HANDLE_PARAMS handle_params;
        memset(&handle_params, 0, sizeof(handle_params));
        handle_params.dwVersion = NVFBC_CREATE_HANDLE_PARAMS_VER;
        handle_params.bExternallyManagedContext = NVFBC_TRUE; // We manage contexts
        handle_params.glxCtx = data.glx_ctx; // Pass the GLX context we created
        handle_params.glxFBConfig = chosen_fb_config;

        ret = nvFBC->nvFBCCreateHandle(&data.nvfbc_session, &handle_params);

        // Reverse engineering for interoperability if needed
        if (ret != NVFBC_SUCCESS) {
            fprintf(stderr, "GLX Capture Thread: nvFBCCreateHandle failed (%s), trying workaround...\n", nvFBC->nvFBCGetLastErrorStr(NULL));
            const uint8_t enable_key[] = { 0xac, 0x10, 0xc9, 0x2e, 0xa5, 0xe6, 0x87, 0x4f, 0x8f, 0x4b, 0xf4, 0x61, 0xf8, 0x56, 0x27, 0xe9 };
            handle_params.privateData = enable_key; 
            handle_params.privateDataSize = 16;
            ret = nvFBC->nvFBCCreateHandle(&data.nvfbc_session, &handle_params);
        }


        pthread_mutex_unlock(&data.session_state_mutex);

        // Unmake GLX context current after nvFBCCreateHandle call
         if (glXMakeContextCurrent_ptr) {
             glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
         } else {
              glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
         }

        if (ret != NVFBC_SUCCESS) {
            fprintf(stderr, "GLX Capture Thread: nvFBCCreateHandle failed after workaround: %s\n", nvFBC->nvFBCGetLastErrorStr(NULL));
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: NVFBC handle created.\n");


        


        // If we reach here, setup was successful
        setup_success = true;

    } while (false); // End of setup block

    if (!setup_success) {
         fprintf(stderr, "GLX Capture Thread: Setup failed. Signaling stop.\n");
         data.stop_threads.store(true); // Signal other threads to stop on setup failure
    }


    // --- Main Capture Loop ---
    // Only run the loop if setup was successful
    if (setup_success) {

        // Check if required EGL extension functions are loaded by main thread
        if (!p_glEGLImageTargetTexture2DOES || !p_eglCreateImageKHR || !p_eglDestroyImageKHR) {
             fprintf(stderr, "GLX Capture Thread: Required EGL extension functions not loaded by main thread! Cannot proceed.\n");
             data.stop_threads.store(true); // Cannot proceed without these functions
        }


        while (!data.stop_threads.load()) {

            // Check for recreate signal (simulating settings change)
            if (data.recreate_session.load()) {
                data.recreate_session.store(false);
                fprintf(stdout, "GLX Capture Thread: Recreating NVFBC session.\n");

                // Destroy existing session if any
                pthread_mutex_lock(&data.session_state_mutex);
                if (data.has_capture_session.load()) {
                    NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                    NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use function pointer
                    if (dest_ret != NVFBC_SUCCESS) {
                        fprintf(stderr, "GLX Capture Thread: nvFBCDestroyCaptureSession failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
                    }
                    data.has_capture_session.store(false);
                }
                pthread_mutex_unlock(&data.session_state_mutex);

                // Create new session with hardcoded settings
                NVFBC_CREATE_CAPTURE_SESSION_PARAMS cap_params = {
                    .dwVersion = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER,
                    .eCaptureType = NVFBC_CAPTURE_TO_GL,
                    .eTrackingType = data.screen_id == -1 ? NVFBC_TRACKING_SCREEN : NVFBC_TRACKING_OUTPUT,
                    .dwOutputId = (uint32_t)data.screen_id, // Cast needed
                    .bWithCursor = NVFBC_FALSE, // Simplicity
                    .bDisableAutoModesetRecovery = NVFBC_TRUE,
                    .bRoundFrameSize = NVFBC_FALSE,
                    .dwSamplingRateMs = data.fps > 0 ? (uint32_t)(1000.0 / data.fps + 0.5) : 33, // ~30 FPS
                    .bPushModel = NVFBC_TRUE, // Non-blocking grab
                    .bAllowDirectCapture = NVFBC_TRUE, // Try direct capture
                };

                pthread_mutex_lock(&data.session_state_mutex);
                NVFBCSTATUS ret = nvFBC->nvFBCCreateCaptureSession(data.nvfbc_session, &cap_params); // Use function pointer
                pthread_mutex_unlock(&data.session_state_mutex);

                if (ret != NVFBC_SUCCESS) {
                    fprintf(stderr, "GLX Capture Thread: nvFBCCreateCaptureSession failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
                    usleep(10000); // Sleep briefly before retrying session creation
                    continue; // Skip to next loop iteration
                }

                // Setup TO_GL
                pthread_mutex_lock(&data.session_state_mutex);
                data.togl_setup_params = (NVFBC_TOGL_SETUP_PARAMS){
                    .dwVersion = NVFBC_TOGL_SETUP_PARAMS_VER,
                    .eBufferFormat = NVFBC_BUFFER_FORMAT_RGBA,
                    .bWithDiffMap = NVFBC_FALSE,
                    .dwDiffMapScalingFactor = 1
                };
                // GLX context MUST be current for this call if bExternallyManagedContext is TRUE
                 if (glXMakeContextCurrent_ptr) {
                     make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
                 } else {
                      make_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); // Direct call
                 }
                 if (!make_current_ok) {
                     fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current for nvFBCToGLSetUp! Fatal.\n");
                     pthread_mutex_unlock(&data.session_state_mutex); // Unlock before exiting
                     data.stop_threads.store(true);
                     continue;
                 }

                ret = nvFBC->nvFBCToGLSetUp(data.nvfbc_session, &data.togl_setup_params); // Use function pointer

                // Unmake GLX context current after setup
                 if (glXMakeContextCurrent_ptr) {
                     glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
                 } else {
                      glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
                 }

                pthread_mutex_unlock(&data.session_state_mutex);


                if (ret != NVFBC_SUCCESS) {
                    fprintf(stderr, "GLX Capture Thread: nvFBCToGLSetUp failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
                    // Destroy the failed session
                    pthread_mutex_lock(&data.session_state_mutex);
                    NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                    nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use function pointer
                    pthread_mutex_unlock(&data.session_state_mutex);
                    data.has_capture_session.store(false);
                    usleep(10000); // Sleep briefly before retrying setup
                    continue; // Skip to next loop iteration
                }
                data.has_capture_session.store(true);
                fprintf(stdout, "GLX Capture Thread: NVFBC session recreated successfully.\n");
            } // end recreate_session check

            if (!data.has_capture_session.load()) {
                // Wait if session creation failed or is pending
                usleep(10000);
                continue; // Skip to next loop iteration
            }

            // --- Grab Frame ---
            // Make GLX context current for NvFBC grab call
            Bool make_glx_current_ok = False;
            if (glXMakeContextCurrent_ptr) {
                 make_glx_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
            } else {
                 make_glx_current_ok = glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx); // Direct call
            }

            if (!make_glx_current_ok) {
                fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current for grab! Fatal.\n");
                data.stop_threads.store(true);
                continue; // Exit loop via stop_threads check
            }

            GLuint glxTex = 0;
            NVFBC_FRAME_GRAB_INFO info = {0};
            NVFBC_TOGL_GRAB_FRAME_PARAMS grab_params = {
                .dwVersion = NVFBC_TOGL_GRAB_FRAME_PARAMS_VER,
                .dwFlags = NVFBC_TOGL_GRAB_FLAGS_NOWAIT | NVFBC_TOGL_GRAB_FLAGS_FORCE_REFRESH, // Use NOWAIT
                .pFrameGrabInfo = &info,
                .dwTimeoutMs = 0 // Timeout is ignored with NOWAIT
            };

            pthread_mutex_lock(&data.session_state_mutex); // Protect NvFBC session handle during call
            NVFBCSTATUS ret = nvFBC->nvFBCToGLGrabFrame(data.nvfbc_session, &grab_params); // Use function pointer
            pthread_mutex_unlock(&data.session_state_mutex);

            bool frame_grabbed = false;
            if (ret == NVFBC_SUCCESS && info.bIsNewFrame == NVFBC_TRUE) {
                 glxTex = data.togl_setup_params.dwTextures[grab_params.dwTextureIndex];
                 if (glxTex != 0) {
                      frame_grabbed = true;
                      // Update dimensions atomically
                      data.frame_width.store(info.dwWidth);
                      data.frame_height.store(info.dwHeight);
                      //fprintf(stdout, "GLX Capture Thread: Grabbed new frame %ux%u, texture %u.\n", info.dwWidth, info.dwHeight, glxTex);
                 } else {
                     fprintf(stderr, "GLX Capture Thread: Grab succeeded but returned texture ID 0.\n");
                 }
            } else if (ret != NVFBC_SUCCESS /*&& ret != NVFBC_ERROR_GL_TIMEOUT*/) { // Don't log timeout as an error
                 // Error other than timeout (timeout is expected with NOWAIT if no new frame)
                 fprintf(stderr, "GLX Capture Thread: nvFBCToGLGrabFrame failed: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
            }

            // Unmake GLX context current after NvFBC grab
            if (glXMakeContextCurrent_ptr) {
                glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
            } else {
                 glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
            }


            // --- Create EGLImage ---
            EGLImageKHR img = EGL_NO_IMAGE_KHR; // Initialize image handle
            bool image_created = false;

            // Only attempt EGLImage creation if a frame was successfully grabbed AND we have the required EGL functions
            if (frame_grabbed && p_eglCreateImageKHR && p_glEGLImageTargetTexture2DOES && data.glx_thread_egl_ctx != EGL_NO_CONTEXT) {



                // Make the shared EGL context current in this thread for EGLImage creation
                 if (!eglMakeCurrent(data.shared_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, data.glx_thread_egl_ctx)) { // Use shared_egl_dpy!
                      EGLint egl_err = eglGetError();
                      fprintf(stderr, "GLX Capture Thread: Failed to make shared EGL context current for EGLImage creation: EGL error %x\n", egl_err);
                 } else {
                     // Bind the texture in the now-current shared EGL context
                     // glBindTexture(GL_TEXTURE_2D, glxTex); // Direct call (GL API)

                     // Create the EGLImage using the shared EGL context and shared EGL display
                     const EGLint attrs[] = {
                         EGL_IMAGE_PRESERVED_KHR, EGL_TRUE,
                         EGL_NONE
                     };
                     img = eglCreateImage(
                         data.shared_egl_dpy,          // Use the SHARED EGL display
                         EGL_NO_CONTEXT,//data.glx_thread_egl_ctx,      // Pass the handle of the shared EGL context (created on shared_egl_dpy)
                         EGL_GL_TEXTURE_2D,            // Target is a 2D GL texture
                         (EGLClientBuffer)(uintptr_t)glxTex, // The GL texture ID
                         NULL
                     );

                     // Unbind texture
                     // glBindTexture(GL_TEXTURE_2D, 0); // Direct call (GL API)

                     // Unmake the shared EGL context current
                     eglMakeCurrent(data.shared_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // Use shared_egl_dpy!

                     if (img == EGL_NO_IMAGE_KHR) {
                          EGLint egl_err = eglGetError(); // Direct call (EGL API)
                          fprintf(stderr, "GLX Capture Thread: Failed to create EGLImage from GLX texture %u (using shared EGL context %p on shared display %p): EGL error %x\n", glxTex, data.glx_thread_egl_ctx, data.shared_egl_dpy, egl_err);
                          image_created = true;
                     } else {

                         // fprintf(stdout, "GLX Capture Thread: Created EGLImage %p from GLX texture %u.\n", img, glxTex);
                     }
                 } // end if eglMakeCurrent success
            } else if (frame_grabbed && (!p_eglCreateImageKHR || !p_glEGLImageTargetTexture2DOES)) {
                 fprintf(stderr, "GLX Capture Thread: Frame grabbed but EGL extension functions are NULL. Cannot create EGLImage.\n");
            } else if (frame_grabbed && data.glx_thread_egl_ctx == EGL_NO_CONTEXT) {
                 fprintf(stderr, "GLX Capture Thread: Frame grabbed but shared EGL context is NULL. Cannot create EGLImage.\n");
            }


            // Push the image handle (EGL_NO_IMAGE_KHR if creation failed or no new frame processed)
            data.imageQueue.push(img);
            // fprintf(stdout, "GLX Capture Thread: Pushed image handle %p to queue.\n", img);

            // Sleep for FPS control or on error/timeout/no frame
            if (image_created) { // Sleep based on FPS only if we successfully created an image
                 if (data.fps > 0) {
                     usleep(1000000 / data.fps); // sleep microseconds
                } else {
                     usleep(1000); // short sleep if FPS is 0 or negative
                }
            }/* else if (ret == NVFBC_ERROR_GL_TIMEOUT) {
                // If timeout, sleep a little less than the target frame time to keep trying
                 if (data.fps > 0) {
                     usleep((uint32_t)(0.8 * (1000000.0 / data.fps))); // sleep ~80% of frame time
                 } else {
                     usleep(5000); // short sleep
                 }
            } */else {
                 // Sleep longer on other errors or if setup failed during loop
                 usleep(10000); // Sleep 10ms
            }

        } // End of capture loop
    }


    // --- Cleanup ---
    fprintf(stdout, "GLX Capture Thread: Cleaning up.\n");

    // Signal the EGL processor thread to stop before cleaning up resources it might use
    data.imageQueue.stop_queue();

    // Ensure NvFBC API functions are loaded before attempting to use them for cleanup
    if (nvFBC && nvFBC->nvFBCDestroyCaptureSession && nvFBC->nvFBCDestroyHandle && nvFBC->nvFBCGetLastErrorStr) {
        // Destroy NVFBC session and handle
        pthread_mutex_lock(&data.session_state_mutex);
        if (data.has_capture_session.load()) {
            NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
            NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use function pointer
            if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture Thread: nvFBCDestroyCaptureSession failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
            data.has_capture_session.store(false);
        }
        // Context might need to be current for DestroyHandle if bExternallyManagedContext was TRUE
        // Making GLX context current before destroying the handle
         if (glXMakeContextCurrent_ptr && data.glx_dpy && data.glx_ctx) { // Check validity
              glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
         } else if (data.glx_dpy && data.glx_ctx) { // Fallback
              glXMakeCurrent(data.glx_dpy, (GLXDrawable)data.glx_pbuffer, data.glx_ctx);
         }


        if (data.nvfbc_session != -1) {
            NVFBC_DESTROY_HANDLE_PARAMS destroy_params = {.dwVersion = NVFBC_DESTROY_HANDLE_PARAMS_VER};
            NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyHandle(data.nvfbc_session, &destroy_params); // Use function pointer
            if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture Thread: nvFBCDestroyHandle failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
            data.nvfbc_session = -1;
        }

        // Unmake GLX context current after destroying handle
         if (glXMakeContextCurrent_ptr && data.glx_dpy) { // Check validity
              glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
         } else if (data.glx_dpy) { // Fallback
              glXMakeCurrent(data.glx_dpy, None, NULL);
         }

        pthread_mutex_unlock(&data.session_state_mutex);
    } else {
         fprintf(stderr, "GLX Capture Thread: NvFBC API functions not fully available for cleanup.\n");
    }


    // Destroy EGL context created in this thread
     if (data.glx_thread_egl_ctx != EGL_NO_CONTEXT && data.shared_egl_dpy != EGL_NO_DISPLAY) { // Use shared_egl_dpy!
         // Ensure it's not current before destroying (shouldn't be based on loop logic, but safe)
          eglMakeCurrent(data.shared_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // Direct call
          if (eglDestroyContext(data.shared_egl_dpy, data.glx_thread_egl_ctx) != EGL_TRUE) { // Direct call
              fprintf(stderr, "GLX Capture Thread: Failed to destroy shared EGL context %p on shared display %p: %d\n", data.glx_thread_egl_ctx, data.shared_egl_dpy, eglGetError());
          } else {
              fprintf(stdout, "GLX Capture Thread: Shared EGL context destroyed.\n");
          }
          data.glx_thread_egl_ctx = EGL_NO_CONTEXT;
     }


    // Destroy GLX context and Pbuffer
    if (data.glx_ctx && data.glx_dpy) { // Check display handle
        glXDestroyContext(data.glx_dpy, data.glx_ctx); // Direct call
        data.glx_ctx = NULL;
         fprintf(stdout, "GLX context destroyed.\n");
    }
    if (data.glx_pbuffer && data.glx_dpy) { // Check display handle
        glXDestroyPbuffer(data.glx_dpy, data.glx_pbuffer); // Direct call
        data.glx_pbuffer = None;
         fprintf(stdout, "GLX Pbuffer destroyed.\n");
    }


    // Close X11 display connection
    if (data.glx_dpy) {
        XCloseDisplay(data.glx_dpy); // Direct call
        data.glx_dpy = NULL;
         fprintf(stdout, "GLX thread X11 display closed.\n");
    }


    fprintf(stdout, "GLX Capture Thread: Exited.\n");
    return NULL;
}

// --- EGL Processor Thread Function ---
static void *egl_processor_thread(void *param)
{
    fprintf(stdout, "EGL Processor Thread: Started.\n");

    // This thread uses the SHARED EGL Display initialized by the main thread.
    EGLDisplay dpy = data.shared_egl_dpy;

    if (dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "EGL Processor Thread: Shared EGL Display is invalid! Fatal.\n");
        data.stop_threads.store(true); // Signal GLX thread too
        // Cleanup happens naturally as the function exits
    } else {
        EGLImageKHR image_for_main = EGL_NO_IMAGE_KHR; // The image handle currently visible to the main thread

        // Ensure EGL destroy function is loaded
        if (!p_eglDestroyImageKHR) {
             fprintf(stderr, "EGL Processor Thread: p_eglDestroyImageKHR is NULL! Cannot destroy images. Fatal.\n");
             data.stop_threads.store(true); // Signal other threads
        } else {
            // Continue processing queue until stop is requested AND the queue is empty
            while (!data.stop_threads.load() || data.imageQueue.size() > 0) {

                // Wait for a new EGLImage from the capture thread
                EGLImageKHR new_image = data.imageQueue.pop(); // Blocking pop, respects stop flag

                if (new_image == EGL_NO_IMAGE_KHR) { // Stop signal received or queue empty on stop
                    if (data.stop_threads.load() && data.imageQueue.size() == 0) {
                        fprintf(stdout, "EGL Processor Thread: Stop signal received and queue is empty.\n");
                        break; // Truly stopping the loop
                    } else {
                         // Received EGL_NO_IMAGE_KHR unexpectedly from the queue (e.g., creation failed in capture thread)
                         fprintf(stderr, "EGL Processor Thread: Received EGL_NO_IMAGE_KHR unexpectedly from queue. Skipping.\n");
                         // We should still process the queue until stop, but don't try to process this invalid image
                         continue;
                    }
                }
                // else fprintf(stdout, "EGL Processor Thread: Popped a new EGLImage handle %p.\n", new_image);


                // Manage the lifecycle of the previous image and swap in the new one
                pthread_mutex_lock(&data.frame_state_mutex); // Protect shared frame state

                // Destroy the old image *before* making the new one available
                // Only destroy if it was a valid handle and we have a valid EGL display/function
                if (image_for_main != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY && p_eglDestroyImageKHR != NULL) {
                    if (p_eglDestroyImageKHR(dpy, image_for_main) != EGL_TRUE) {
                        fprintf(stderr, "EGL Processor Thread: Failed to destroy old EGLImage %p on display %p: %d\n", image_for_main, dpy, eglGetError()); // Direct call
                    } // else fprintf(stdout, "EGL Processor Thread: Destroyed old EGLImage %p.\n", image_for_main);
                }

                // Make the new image available to the main thread
                data.current_render_image = new_image;
                image_for_main = new_image; // Keep track of the image we just made available

                // Signal the main thread that a new image is ready
                // Dimensions are updated atomically by the GLX thread, main thread reads them.
                data.new_image_available.store(true);
                //fprintf(stdout, "EGL Processor Thread: Signaled new image available.\n");

                pthread_mutex_unlock(&data.frame_state_mutex); // Unlock state mutex

            } // End of processor loop

            // --- Cleanup after loop ---
            fprintf(stdout, "EGL Processor Thread: Cleaning up queue and last image.\n");

            // Destroy any remaining image handle we were managing
            pthread_mutex_lock(&data.frame_state_mutex);
            if (image_for_main != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY && p_eglDestroyImageKHR != NULL) {
                if (p_eglDestroyImageKHR(dpy, image_for_main) != EGL_TRUE) {
                    fprintf(stderr, "EGL Processor Thread: Failed to destroy final EGLImage %p on display %p during cleanup: %d\n", image_for_main, dpy, eglGetError()); // Direct call
                } // else fprintf(stdout, "EGL Processor Thread: Destroyed final EGLImage %p during cleanup.\n", image_for_main);
            }
            data.current_render_image = EGL_NO_IMAGE_KHR; // Reset state
            data.new_image_available.store(false);
            pthread_mutex_unlock(&data.frame_state_mutex);

            // Clear any images left in the queue (should be empty if GLX thread finished correctly, but defensive)
            data.imageQueue.clear(dpy);
        } // end if p_eglDestroyImageKHR is valid
    }


    fprintf(stdout, "EGL Processor Thread: Exited.\n");
    return NULL;
}


// --- Main Application Functions ---

// Placeholder render function (draws a simple textured quad)
// Assumes main_egl_ctx is current.
void render_texture(GLuint tex, uint32_t width, uint32_t height) {
    // Basic GL calls linked from libGL/libGLX
    // Need a simple shader program to draw the texture. This requires GLSL compilation and linking.
    // For this minimal example, we'll skip actual shader creation and just assume a simple shader exists
    // and is made current *outside* this function, or use fixed-function pipeline if available (unlikely with modern GL).
    // A real implementation needs shader setup.
    // Assuming a simple pass-through vertex shader and texture sampling fragment shader are bound...

    if (tex == 0 || width == 0 || height == 0) return;

    // Minimal VAO/VBO for a quad
    static GLuint vao = 0;
    static GLuint vbo = 0;
    static bool vao_vbo_initialized = false;

    if (!vao_vbo_initialized) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        float vertices[] = {
            // Positions        // Texture Coords (adjust if Y needs flipping - NvFBC TO_GL usually Y-flipped)
            -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // Bottom-left (maps to top-left of source image)
             1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // Bottom-right (maps to top-right)
             1.0f,  1.0f, 0.0f, 1.0f, 0.0f, // Top-right (maps to bottom-right)
            -1.0f,  1.0f, 0.0f, 0.0f, 0.0f  // Top-left (maps to bottom-left)
        };

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // Texture coord attribute (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        vao_vbo_initialized = true;
    }

    // --- Rendering ---
    // NOTE: A real application needs to bind a shader program before rendering.
    // This sample assumes a program is bound externally or implicitly (e.g. fixed-function)
    // which is unlikely with a core profile context. Add shader code here!

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Grayish background
    glClear(GL_COLOR_BUFFER_BIT);

    // Activate texture unit (e.g., 0) and bind the texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    // If you had a shader uniform for the texture sampler, set it here:
    // GLint tex_uniform_loc = glGetUniformLocation(your_shader_program, "texture_sampler");
    // if (tex_uniform_loc != -1) glUniform1i(tex_uniform_loc, 0); // Use texture unit 0

    // Draw the quad
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // Draw the quad using TRIANGLE_FAN
    glBindVertexArray(0);

    // Clean up bindings
    glBindTexture(GL_TEXTURE_2D, 0);
    // glUseProgram(0); // Unbind shader program (if used)


    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "GL error during render_texture: %x\n", err);
    }

    // Swap buffers to show the rendered frame
     eglSwapBuffers(data.shared_egl_dpy, data.main_egl_surface);
}


// Helper to check GL extensions
bool check_gl_ext(const char *name) {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
        fprintf(stderr, "Cannot check GL extensions, no EGL context current.\n");
        return false;
    }

    GLint ext_count = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &ext_count);
    if (ext_count > 0) {
        for (GLint i = 0; i < ext_count; ++i) {
            const GLubyte *ext = glGetStringi(GL_EXTENSIONS, i);
            if (!ext) {
                fprintf(stderr, "Could not get OpenGL extension #%i\n", i);
            } else if (!strcmp((const char *)ext, name)) {
                return true;
            }
        }
    } else {
        const char *extensions = (const char *)glGetString(GL_EXTENSIONS);
        if (extensions) {
            return strstr(extensions, name) != NULL;
        } else {
            fprintf(stderr, "Could not get OpenGL extensions string\n");
        }
    }
    return false;
}

// Helper to check EGL extensions
bool check_egl_ext(EGLDisplay dpy, const char *name) {
    if (dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "Cannot check EGL extensions, invalid display.\n");
        return false;
    }
    const char *extensions = eglQueryString(dpy, EGL_EXTENSIONS);
    if (!extensions) {
        fprintf(stderr, "Could not get EGL extensions string\n");
        return false;
    }
    // printf("EXTENSIONS:\n%s\n",extensions); // Uncomment to see supported EGL extensions
    return strstr(extensions, name) != NULL;
}


int main() {
    fprintf(stdout, "Minimal NvFBC EGL/GLX Test (Linux)\n");

    // --- 0. Initialize shared data mutexes ---
    pthread_mutex_init(&data.session_state_mutex, NULL);
    pthread_mutex_init(&data.frame_state_mutex, NULL);

    bool main_setup_success = false;

    // Use separate flags for each resource created during setup
    
    bool main_x_dpy_opened = false;
    bool main_x_win_created = false;
    bool shared_egl_dpy_initialized = false;
    bool main_egl_config_chosen = false;
    bool main_egl_ctx_created = false;
    bool main_egl_surface_created = false;
    pthread_t glx_thread_handle = 0;
    pthread_t egl_processor_thread_handle = 0;


    do { // Use a do-while(false) loop to allow using break for early exit during setup
        // --- Load NvFBC Library Dynamically ---
        


        // --- 1. Setup Main Thread EGL/X11 (Simulating OBS Context) ---
        fprintf(stdout, "Setting up main thread EGL/X11...\n");

        data.main_x_dpy = XOpenDisplay(NULL); // Direct call
        if (!data.main_x_dpy) {
            fprintf(stderr, "Error: Failed to open main X11 display.\n");
            break; // Exit setup block
        }
        main_x_dpy_opened = true;
        int main_screen = XDefaultScreen(data.main_x_dpy); // Direct call
        Window main_root_win = XRootWindow(data.main_x_dpy, main_screen); // Direct call

        // Create a dummy X11 window for EGL surface/context
        data.main_x_win = XCreateSimpleWindow(data.main_x_dpy, main_root_win, 0, 0, 640, 480, 0, 0, 0); // Direct call
        if (data.main_x_win == 0) {
             fprintf(stderr, "Error: Failed to create main X11 window.\n");
             break; // Exit setup block
        }
        main_x_win_created = true;
        XMapWindow(data.main_x_dpy, data.main_x_win); // Map it to show the window
        XFlush(data.main_x_dpy); // Ensure window is mapped

        // Get EGL display for the main X11 display (this will be the shared display handle)
        data.shared_egl_dpy = eglGetDisplay((EGLNativeDisplayType)data.main_x_dpy); // Direct call
        if (data.shared_egl_dpy == EGL_NO_DISPLAY) {
            fprintf(stderr, "Error: Failed to get main EGL Display.\n");
            break; // Exit setup block
        }

        EGLint major_egl, minor_egl;
        if (!eglInitialize(data.shared_egl_dpy, &major_egl, &minor_egl)) { // Direct call
            fprintf(stderr, "Error: Failed to initialize main EGL Display %p: %d\n", data.shared_egl_dpy, eglGetError()); // Direct call
            break; // Exit setup block
        }
        shared_egl_dpy_initialized = true;
        fprintf(stdout, "Main thread EGL Display %p initialized (version %d.%d).\n", data.shared_egl_dpy, major_egl, minor_egl);

        // Check required EGL extensions on the shared display
        if (!check_egl_ext(data.shared_egl_dpy, "EGL_KHR_image_base")) {
            fprintf(stderr, "Error: Required EGL extension 'EGL_KHR_image_base' not supported on main display %p.\n", data.shared_egl_dpy); break;
        }
        if (!check_egl_ext(data.shared_egl_dpy, "EGL_KHR_gl_texture_2D_image")) {
             fprintf(stderr, "Error: Required EGL extension 'EGL_KHR_gl_texture_2d_image' not supported on main display %p.\n", data.shared_egl_dpy); break;
        }
         fprintf(stdout, "Required EGL extensions found on main display.\n");

        // Load required EGL extension functions using the shared display
        p_eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
        p_eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
        p_glEGLImageTargetTexture2DOES = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");

        if (!p_eglCreateImageKHR || !p_eglDestroyImageKHR || !p_glEGLImageTargetTexture2DOES) {
             fprintf(stderr, "Error: Required EGL extension functions not found (eglCreateImageKHR=%p, eglDestroyImageKHR=%p, glEGLImageTargetTexture2DOES=%p) via eglGetProcAddress. Fatal.\n",
                     p_eglCreateImageKHR, p_eglDestroyImageKHR, p_glEGLImageTargetTexture2DOES);
             break; // Exit setup block
        }
        fprintf(stdout, "Required EGL extension functions loaded via eglGetProcAddress.\n");


        // Choose an EGL config for the main context and sharing
        const EGLint config_attrs[] = {
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT, // Need window bit for main surface
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, // Request OpenGL support
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
            EGL_NONE
        };
        EGLConfig configs[1];
        EGLint num_configs;
        if (!eglChooseConfig(data.shared_egl_dpy, config_attrs, configs, 1, &num_configs) || num_configs == 0) { // Direct call
            fprintf(stderr, "Error: Failed to choose EGL config for main context.\n");
            break; // Exit setup block
        }
        data.shared_egl_config = configs[0];
        main_egl_config_chosen = true;
        fprintf(stdout, "Main thread EGL config chosen.\n");

        // Create main thread EGL context
        // Try modern OpenGL first, fallback to GLES2
        static const EGLint context_attribs_gl45[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 5, EGL_NONE };
        static const EGLint context_attribs_gl33[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
        static const EGLint context_attribs_gles2[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };

        data.main_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, EGL_NO_CONTEXT, context_attribs_gl33); // Direct call
         if (data.main_egl_ctx == EGL_NO_CONTEXT) {
              fprintf(stdout, "Error: Failed to create main EGL context (4.5+), trying 3.3.\n");
              data.main_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, EGL_NO_CONTEXT, context_attribs_gl33);
         }
         if (data.main_egl_ctx == EGL_NO_CONTEXT) {
              fprintf(stdout, "Error: Failed to create main EGL context (3.3), trying GLES 2.\n");
               data.main_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, EGL_NO_CONTEXT, context_attribs_gles2);
               if (data.main_egl_ctx == EGL_NO_CONTEXT) {
                    fprintf(stderr, "Error: Failed to create any main EGL context: %d.\n", eglGetError());
                    break; // Exit setup block
               }
         }
        main_egl_ctx_created = true;
        fprintf(stdout, "Main thread EGL context created (%p).\n", data.main_egl_ctx);

        // Create EGL surface for the main window
        data.main_egl_surface = eglCreateWindowSurface(data.shared_egl_dpy, data.shared_egl_config, (EGLNativeWindowType)data.main_x_win, NULL); // Direct call
         if (data.main_egl_surface == EGL_NO_SURFACE) {
             fprintf(stderr, "Error: Failed to create main EGL window surface: %d.\n", eglGetError());
             break; // Exit setup block
         }
        main_egl_surface_created = true;
        fprintf(stdout, "Main thread EGL surface created (%p).\n", data.main_egl_surface);

        // Make main context current (needed for GL calls like texture creation and extension checks)
        if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) {
            fprintf(stderr, "Error: Failed to make main EGL context current: %d\n", eglGetError());
            break; // Exit setup block
        }
        fprintf(stdout, "Main thread EGL context is current.\n");
        printf("Main Context OpenGL Version: %s\n",glGetString(GL_VERSION)); // Direct call
        printf("Main Context OpenGL Renderer: %s\n",glGetString(GL_RENDERER)); // Direct call


        // Check required GL extension (GL_OES_EGL_image) in the main context
        if (!check_gl_ext("GL_OES_EGL_image")) {
             fprintf(stdout, "Note: GL_OES_EGL_image extension check failed in main context. Binding EGLImage to GL texture relies on implementation support.\n");
        } else {
            fprintf(stdout, "GL_OES_EGL_image extension seems available in main context.\n");
        }

        // Signal that main EGL setup is complete so GLX thread can use the shared display/context
        data.main_egl_setup_complete.store(true);
        fprintf(stdout, "Main thread EGL setup signaled as complete.\n");


        // --- 2. Create Threads ---
        fprintf(stdout, "Creating threads...\n");

        int ret_glx = pthread_create(&glx_thread_handle, NULL, glx_capture_thread, NULL);
        if (ret_glx != 0) {
            fprintf(stderr, "Error: Failed to create GLX capture thread: %s\n", strerror(ret_glx));
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread created.\n");


        int ret_egl = pthread_create(&egl_processor_thread_handle, NULL, egl_processor_thread, NULL);
        if (ret_egl != 0) {
            fprintf(stderr, "Error: Failed to create EGL processor thread: %s\n", strerror(ret_egl));
            data.stop_threads.store(true); // Signal GLX thread too
            // Attempt to join GLX thread before main cleanup
            if (glx_thread_handle != 0) {
                 fprintf(stdout, "Joining GLX thread after EGL thread creation failure...\n");
                 pthread_join(glx_thread_handle, NULL);
                 glx_thread_handle = 0; // Mark as joined
            }
            break; // Exit setup block
        }
        fprintf(stdout, "EGL Processor Thread created.\n");

        // If we reached here, all main setup was successful and threads are running
        main_setup_success = true;

    } while (false); // End of setup block


    // --- 3. Main Loop (Simulate OBS Render Loop) ---
    if (main_setup_success) {
        fprintf(stdout, "Entering main render loop. Close window to stop.\n");

        // Atom for window close message
        Atom wm_delete_window;
        wm_delete_window = XInternAtom(data.main_x_dpy, "WM_DELETE_WINDOW", False); // Direct call
        XSetWMProtocols(data.main_x_dpy, data.main_x_win, &wm_delete_window, 1); // Direct call

        // Make the main EGL context current for the loop (it should already be, but ensure)
        if (!(eglGetCurrentContext() == data.main_egl_ctx && eglGetCurrentSurface(EGL_DRAW) == data.main_egl_surface)) {
             if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) {
                  fprintf(stderr, "Main Thread: Failed to make EGL context current for main loop: %d. Rendering will not work.\n", eglGetError());
             } else {
                 fprintf(stdout, "Main Thread: EGL context current for render loop.\n");
             }
        }


        // Minimal render loop (checks for updates and binds texture)
        while (!data.stop_threads.load()) {
            // Poll for new image availability signal
            bool new_image_ready = data.new_image_available.exchange(false); // Atomically get and reset

            if (new_image_ready) {
                // fprintf(stdout, "Main Thread: New image available signal received.\n");

                // Ensure context is current for GL calls before creating/deleting textures
                 if (!(eglGetCurrentContext() == data.main_egl_ctx && eglGetCurrentSurface(EGL_DRAW) == data.main_egl_surface)) {
                     if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) {
                         fprintf(stderr, "Main Thread: Failed to make main EGL context current for texture operations: %d\n", eglGetError());
                         // Cannot proceed with this image if context fails
                         new_image_ready = false; // Treat as if no new image processed successfully
                     } // else { fprintf(stdout, "Main Thread: Made EGL context current for texture operations.\n"); }
                 }

                if (new_image_ready) { // Proceed only if context is current
                     pthread_mutex_lock(&data.frame_state_mutex); // Lock to read shared image handle
                     EGLImageKHR image_to_bind = data.current_render_image;
                     pthread_mutex_unlock(&data.frame_state_mutex); // Unlock after reading

                     uint32_t width = data.frame_width.load(); // Read dimensions atomically
                     uint32_t height = data.frame_height.load();

                     if (image_to_bind != EGL_NO_IMAGE_KHR) {
                          // fprintf(stdout, "Main Thread: Processing new EGLImage %p (%ux%u).\n", image_to_bind, width, height);

                         // Destroy the old main texture if it exists or dimensions changed
                         GLint current_tex_width = 0, current_tex_height = 0;
                         if (data.main_render_tex != 0) {
                             glBindTexture(GL_TEXTURE_2D, data.main_render_tex);
                             glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &current_tex_width);
                             glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &current_tex_height);
                             glBindTexture(GL_TEXTURE_2D, 0);
                         }

                         if (data.main_render_tex != 0 && (current_tex_width != (GLint)width || current_tex_height != (GLint)height)) {
                             glDeleteTextures(1, &data.main_render_tex);
                             data.main_render_tex = 0;
                             fprintf(stdout, "Main Thread: Destroyed old render texture %u due to size change (%dx%d -> %ux%u).\n", data.main_render_tex, current_tex_width, current_tex_height, width, height);
                         }

                         // Create new main texture if needed and dimensions are valid
                        if (data.main_render_tex == 0 && width > 0 && height > 0) {
                             glGenTextures(1, &data.main_render_tex);
                             if (data.main_render_tex == 0) {
                                 fprintf(stderr, "Main Thread: Failed to create main render texture.\n");
                             } else {
                                  glBindTexture(GL_TEXTURE_2D, data.main_render_tex);
                                  // Allocate storage for the texture (immutable storage is better if available, e.g., glTexStorage2D)
                                  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0); // No mipmaps needed
                                  glBindTexture(GL_TEXTURE_2D, 0);
                                  fprintf(stdout, "Main Thread: Created new render texture %u (%ux%u).\n", data.main_render_tex, width, height);
                             }
                        }

                        // If we have a valid texture and image, bind the image to the texture
                        if (data.main_render_tex != 0 && p_glEGLImageTargetTexture2DOES != NULL) {
                            glBindTexture(GL_TEXTURE_2D, data.main_render_tex);
                            p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image_to_bind); // Call via pointer
                            glBindTexture(GL_TEXTURE_2D, 0);

                            GLenum glerr = glGetError();
                            if (glerr != GL_NO_ERROR) {
                                 fprintf(stderr, "Main Thread: glEGLImageTargetTexture2DOES GL error after binding EGLImage %p to texture %u: %x\n", image_to_bind, data.main_render_tex, glerr);
                                 // Binding failed, texture might be invalid now. Destroy it.
                                 glDeleteTextures(1, &data.main_render_tex);
                                 data.main_render_tex = 0;
                                 fprintf(stderr, "Main Thread: Destroyed render texture due to binding error.\n");
                            } // else { fprintf(stdout, "Main Thread: Bound EGLImage %p to texture %u.\n", image_to_tex_bind, data.main_render_tex); }
                        } else if (data.main_render_tex != 0 && p_glEGLImageTargetTexture2DOES == NULL) {
                             fprintf(stderr, "Main Thread: p_glEGLImageTargetTexture2DOES is NULL. Cannot bind EGLImage to texture %u.\n", data.main_render_tex);
                        }
                     } else {
                         // Received EGL_NO_IMAGE_KHR (but not the stop sentinel) - indicates a problem.
                         // Keep the last good frame, log a warning.
                         fprintf(stderr, "Main Thread: Received signal but image handle is EGL_NO_IMAGE_KHR from processor. Skipping texture update.\n");
                     }
                } // end if new_image_ready and context is current
            } // end if new_image_ready

            // --- Render ---
            // Ensure context is current for rendering
             if (main_egl_ctx_created && data.shared_egl_dpy != EGL_NO_DISPLAY) { // Check display and context creation flag
                  if (!(eglGetCurrentContext() == data.main_egl_ctx && eglGetCurrentSurface(EGL_DRAW) == data.main_egl_surface)) {
                      if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) {
                           fprintf(stderr, "Main Thread: Failed to make main EGL context current for rendering: %d\n", eglGetError());
                           // Rendering won't work this frame
                      } // else { fprintf(stdout, "Main Thread: Made EGL context current for rendering.\n"); }
                  }
             }

            // Perform the actual rendering using the texture (if valid and context is current)
            if (eglGetCurrentContext() == data.main_egl_ctx && eglGetCurrentSurface(EGL_DRAW) == data.main_egl_surface) {
                 if (data.main_render_tex != 0) {
                     // Check if texture size matches current window size (simple check, needs refinement for real app)
                     // For this example, we render the texture at [-1, 1] regardless of window size
                     render_texture(data.main_render_tex, data.frame_width.load(), data.frame_height.load()); // Draw the texture
                 } else {
                     // Clear window if no texture to show
                     glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
                     glClear(GL_COLOR_BUFFER_BIT);
                     eglSwapBuffers(data.shared_egl_dpy, data.main_egl_surface);
                 }
            } else {
                 // If context isn't current, can't render or swap.
                 // fprintf(stdout, "Main Thread: Skipping render, EGL context not current.\n");
            }


            // Handle X events for window close (simple poll)
            XEvent event;
            while (main_x_dpy_opened && data.main_x_dpy && XPending(data.main_x_dpy)) {
                XNextEvent(data.main_x_dpy, &event);
                if (event.type == ClientMessage) {
                     if ((Atom)event.xclient.data.l[0] == wm_delete_window) {
                          fprintf(stdout, "Window close requested.\n");
                          data.stop_threads.store(true);
                     }
                }
                 // Add other event handling here if needed (resize, expose, etc.)
            }

             usleep(5000); // Short sleep to avoid excessive CPU usage when no events/updates

        } // End main loop

        fprintf(stdout, "Main loop finished.\n");
    } // End if (main_setup_success)


    // --- 4. Cleanup ---
    fprintf(stdout, "Cleaning up main thread resources...\n");

    // Signal threads to stop and join - do this unconditionally if threads might have been created
    data.stop_threads.store(true);
    data.imageQueue.stop_queue(); // Unblock EGL processor thread pop

    // Join threads if they were successfully created
    if (glx_thread_handle != 0) {
         fprintf(stdout, "Joining GLX thread...\n");
         pthread_join(glx_thread_handle, NULL);
    }
    if (egl_processor_thread_handle != 0) {
         fprintf(stdout, "Joining EGL processor thread...\n");
         pthread_join(egl_processor_thread_handle, NULL);
    }


    // Destroy main thread GL texture IF EGL context was created and valid display
    if (main_egl_ctx_created && data.shared_egl_dpy != EGL_NO_DISPLAY) {
         // Ensure context is current for GL calls
         if (eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) {
             if (data.main_render_tex != 0) {
                 glDeleteTextures(1, &data.main_render_tex);
                 data.main_render_tex = 0;
                 fprintf(stdout, "Main thread render texture destroyed.\n");
             }
             // Unmake context current
             eglMakeCurrent(data.shared_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
         } else {
             fprintf(stderr, "Failed to make main EGL context current for texture cleanup. Texture might leak.\n");
         }
    }

    // Destroy EGL surface IF created and valid display
    if (main_egl_surface_created && data.shared_egl_dpy != EGL_NO_DISPLAY) {
        eglDestroySurface(data.shared_egl_dpy, data.main_egl_surface);
        data.main_egl_surface = EGL_NO_SURFACE;
        fprintf(stdout, "Main thread EGL surface destroyed.\n");
    }

    // Destroy EGL context IF created and valid display
    if (main_egl_ctx_created && data.shared_egl_dpy != EGL_NO_DISPLAY) {
        eglDestroyContext(data.shared_egl_dpy, data.main_egl_ctx);
        data.main_egl_ctx = EGL_NO_CONTEXT;
        fprintf(stdout, "Main thread EGL context destroyed.\n");
    }

    // Terminate EGL display IF initialized
    if (shared_egl_dpy_initialized && data.shared_egl_dpy != EGL_NO_DISPLAY) {
        eglTerminate(data.shared_egl_dpy);
        data.shared_egl_dpy = EGL_NO_DISPLAY;
        fprintf(stdout, "Main thread EGL terminated.\n");
    }

    // Close NvFBC Library Handle IF loaded
    if (data.nvfbc_lib_loaded) {
        fprintf(stdout, "Closing NvFBC library handle.\n");
        os_dlclose(data.nvfbc_lib_handle);
        data.nvfbc_lib_handle = NULL;
    }

    // Destroy main X11 window IF created and valid display
     if (main_x_win_created && data.main_x_dpy) {
          XDestroyWindow(data.main_x_dpy, data.main_x_win);
          data.main_x_win = 0;
          fprintf(stdout, "Main thread X11 window destroyed.\n");
     }

    // Close X11 display IF opened
    if (main_x_dpy_opened && data.main_x_dpy) {
        XCloseDisplay(data.main_x_dpy);
        data.main_x_dpy = NULL;
         fprintf(stdout, "Main thread X11 display closed.\n");
    }

    // Destroy mutexes (initialized unconditionally)
    pthread_mutex_destroy(&data.session_state_mutex);
    pthread_mutex_destroy(&data.frame_state_mutex);

    fprintf(stdout, "Cleanup complete. Exiting.\n");
    return 0;
}