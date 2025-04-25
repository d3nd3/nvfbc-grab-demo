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
             // If T could be other types, a boolean pair or optional<T> might be better.
             // Assuming T is EGLImageKHR for this specific use case.
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
            if (img != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY) {
                 // Using eglDestroyImageKHR directly as it's part of EGL, not NvFBC
                 if (p_eglDestroyImageKHR(dpy, img) != EGL_TRUE) {
                     fprintf(stderr, "Error destroying queued EGLImage during clear: %d\n", eglGetError());
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
    pthread_mutex_t session_state_mutex; // Protects nvfbc_session, has_capture_session, togl_setup_params, togl_setup_params

    // Capture Thread (GLX) handles
    Display *glx_dpy = NULL;
    GLXContext glx_ctx = NULL;
    GLXPbuffer glx_pbuffer = None;
    EGLDisplay glx_thread_local_egl_dpy = EGL_NO_DISPLAY; // EGL display local to this thread

    // EGL Data (shared display, used by Processor and Main)
    EGLDisplay shared_egl_dpy = EGL_NO_DISPLAY;
    EGLConfig shared_egl_config = NULL; // Could be useful for creating shared contexts if needed

    // EGLImage Shared Queue (Capture produces, Processor consumes)
    SharedQueue<EGLImageKHR> imageQueue;

    // Shared state between EGL Processor Thread and Main Render Thread
    // EGL Processor thread updates these, Render thread reads.
    EGLImageKHR current_render_image = EGL_NO_IMAGE_KHR; // Protected by frame_state_mutex
    std::atomic<bool> new_image_available = false; // Signaled by EGL Processor
    std::atomic<uint32_t> frame_width = 0; // Protected by frame_state_mutex
    std::atomic<uint32_t> frame_height = 0; // Protected by frame_state_mutex
    pthread_mutex_t frame_state_mutex; // Protects current_render_image, frame_width, frame_height

    // Main Thread (EGL) handles - for minimal rendering
    Display *main_x_dpy = NULL; // The X11 display used by the main thread
    Window main_x_win = 0;      // The X11 window used by the main thread
    EGLContext main_egl_ctx = EGL_NO_CONTEXT;
    GLuint main_render_tex = 0; // Texture in main thread's GL context
    EGLSurface main_egl_surface = EGL_NO_SURFACE; // EGL surface for the window

    // Settings (minimal, hardcoded or read once)
    int screen_id = -1; // -1 for entire desktop
    int fps = 30;
};

static struct shared_data data;


// --- GLX Capture Thread Function ---
static void *glx_capture_thread(void *param)
{
    fprintf(stdout, "GLX Capture Thread: Started.\n");

    // Get pointer to the shared NVFBC API function list
    NVFBC_API_FUNCTION_LIST *nvFBC = &data.nvFBC_api;

    bool setup_success = false; // Flag to track overall setup success

    PFNGLXMAKECONTEXTCURRENTPROC glXMakeContextCurrent_ptr;

    // --- 1. Open X11 Display and create GLX context/Pbuffer ---
    do { // Use a do-while(false) loop to allow using break for early exit during setup
        data.glx_dpy = XOpenDisplay(NULL); // Direct call
        if (!data.glx_dpy) {
            fprintf(stderr, "GLX Capture Thread: Failed to open X11 display.\n");
            break; // Exit setup block
        }
        int screen = DefaultScreen(data.glx_dpy); // Direct call

        
        #if 1
        // Initialize thread-local EGL display (using its X display)
        data.glx_thread_local_egl_dpy = eglGetDisplay((EGLNativeDisplayType)data.glx_dpy); // Direct call
        if (data.glx_thread_local_egl_dpy == EGL_NO_DISPLAY) {
            fprintf(stderr, "GLX Capture Thread: Failed to get thread-local EGL Display.\n");
            break; // Exit setup block
        }
        EGLint major, minor;
        if (!eglInitialize(data.glx_thread_local_egl_dpy, &major, &minor)) { // Direct call
            fprintf(stderr, "GLX Capture Thread: Failed to initialize thread-local EGL Display: %d\n", eglGetError()); // Direct call
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: Initialized thread-local EGL Display (version %d.%d).\n", major, minor);
        #endif
        
        #define USE_PBUFFER
        // #define USE_LEGACY

        //legacy actually uses opengl 4.6 lol

        #ifndef USE_LEGACY

        #ifdef USE_PBUFFER
        // Get GLXFBConfig for modern context and Pbuffer
        
        int visual_attribs[] = {
            GLX_X_RENDERABLE, True, //required and is default
            GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
            GLX_DOUBLEBUFFER, True,
            // GLX_DEPTH_SIZE, 0,
            // GLX_STENCIL_SIZE, 0,
            None
        };

        #else
        int visual_attribs[] = {
            GLX_X_RENDERABLE, True,
            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 0,//ALPHA=0 if WINDOW.
            GLX_DOUBLEBUFFER, True,
            None
        };
        #endif

        int fbcount;
        GLXFBConfig *configs;
        configs = glXChooseFBConfig(data.glx_dpy, screen, visual_attribs, &fbcount); // Direct call
        if (fbcount == 0 || !configs) {
            fprintf(stderr, "GLX Capture Thread: Failed to get GLXFBConfig.\n");
            break; // Exit setup block
        }
        //copies the config.
        GLXFBConfig chosen = configs[0];
        XFree(configs);
        

        fprintf(stdout, "GLX Capture Thread: Found %d FBConfigs, using first one.\n", fbcount);

        // Create GLX context (modern core profile preferred)
        static const int context_attribs[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3, GLX_CONTEXT_MINOR_VERSION_ARB, 1,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        /*
            Modern core profile not work if set to 4.6

            GLXContext glXCreateContextAttribsARB(
                    Display *dpy,
                    GLXFBConfig config,
                    GLXContext share_context,
                    Bool direct,
                    const int *attrib_list);

            Shared contexts allow for efficient resource utilization and can facilitate multithreaded rendering workflows.

            For an OpenGL or OpenGL ES context, respecification always results in orphaning of the EGLImage, and may
            also include allocation of additional memory for the respecified resource and/or
            copying of the EGLImage pixel data
        */
 
       
        PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB_ptr = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");
        if (glXCreateContextAttribsARB_ptr) {
            data.glx_ctx = glXCreateContextAttribsARB_ptr(data.glx_dpy, chosen, NULL, True, context_attribs);
        } else {
             fprintf(stdout, "GLX Capture Thread: glXCreateContextAttribsARB not found, trying direct call.\n");
             // Attempt direct call if header provided it
             data.glx_ctx = glXCreateContextAttribsARB(data.glx_dpy, chosen, NULL, True, context_attribs);
        }
        #else //USE_LEGACY
        #ifdef USE_PBUFFER

            
            int visual_attribs[] = {
                GLX_X_RENDERABLE, True,
                GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                GLX_RENDER_TYPE, GLX_RGBA_BIT,
                GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
                GLX_RED_SIZE, 8,
                GLX_GREEN_SIZE, 8,
                GLX_BLUE_SIZE, 8,
                GLX_ALPHA_SIZE, 0, //this has to be 0 if using LEGACY 
                GLX_DEPTH_SIZE, 24,
                GLX_STENCIL_SIZE, 8,
                None
            };
            

        #else 
        int visual_attribs[] = {
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
            // TODO:
            //GLX_BIND_TO_TEXTURE_RGBA_EXT, 1,
            //GLX_BIND_TO_TEXTURE_TARGETS_EXT, GLX_TEXTURE_2D_BIT_EXT,
            GLX_DOUBLEBUFFER, True,
            GLX_RED_SIZE, 8,
            GLX_GREEN_SIZE, 8,
            GLX_BLUE_SIZE, 8,
            GLX_ALPHA_SIZE, 0, //this has to be 0 if using LEGACY 
            GLX_DEPTH_SIZE, 0,
            None,
        };
        #endif //USE_PBUFFER

        GLXFBConfig chosen;
        #if 1
        int fbcount;
        GLXFBConfig *configs;
        configs = glXChooseFBConfig(data.glx_dpy, screen, visual_attribs, &fbcount); // Direct call
        if (fbcount == 0 || !configs) {
            fprintf(stderr, "GLX Capture Thread: Failed to get GLXFBConfig.\n");
            break; // Exit setup block
        }
        //copies the config.
        chosen = configs[0];
        XFree(configs);
        #endif //USE_LEGACY
       
        if (!data.glx_ctx) {
             fprintf(stdout, "GLX Capture Thread: Modern context creation failed, trying legacy.\n");

             #if 0
             int legacy_attrs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };
             XVisualInfo *vi = glXChooseVisual(data.glx_dpy, screen, legacy_attrs); // Direct call
             if (vi) {
                data.glx_ctx = glXCreateContext(data.glx_dpy, vi, NULL, True); // Direct call
                XFree(vi); // Direct call
             }
             #else
             //gsr uses this
             data.glx_ctx = glXCreateNewContext(data.glx_dpy, chosen, GLX_RGBA_TYPE, NULL, True);
             #endif
        }
        #endif //USE_LEGACY

        int mmajor, mminor;
        if (glXQueryVersion(data.glx_dpy, &mmajor, &mminor)) {
            printf("GLX Version: %d.%d\n", mmajor, mminor);
        } else {
            fprintf(stderr, "Error: glXQueryVersion failed.\n");
        }



        if (!data.glx_ctx) {
            fprintf(stderr, "GLX Capture Thread: Failed to create any GLX context.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: GLX context created.\n");

        
        #ifdef USE_PBUFFER
        // Create GLX Pbuffer (dummy surface to make context current)
        static const int pbuffer_attribs[] = { GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, None };
         
        // Use function pointer if available, fallback to direct call
        PFNGLXCREATEPBUFFERPROC glXCreatePbuffer_ptr = (PFNGLXCREATEPBUFFERPROC)glXGetProcAddress((const GLubyte*)"glXCreatePbuffer");
        if (glXCreatePbuffer_ptr) {
         data.glx_pbuffer = glXCreatePbuffer_ptr(data.glx_dpy, chosen, pbuffer_attribs);
        } else {
          fprintf(stdout, "GLX Capture Thread: glXCreatePbuffer not found, trying direct call.\n");
         data.glx_pbuffer = glXCreatePbuffer(data.glx_dpy, chosen, pbuffer_attribs); // Direct call
        }

        if (!data.glx_pbuffer) {
          fprintf(stderr, "GLX Capture Thread: Failed to create GLX Pbuffer surface.\n");
        } else {
          fprintf(stdout, "GLX Capture Thread: GLX Pbuffer created.\n");
        }
        #else
            // depth=CopyFromParent
            Window dummy_win = XCreateWindow(data.glx_dpy, DefaultRootWindow(data.glx_dpy), 0, 0, 16, 16, 0, CopyFromParent, InputOutput, CopyFromParent, 0, NULL);
            if(!dummy_win) {
                fprintf(stderr, "failed to create gl window\n");
                break;
            }
        #endif
         
        /*
            This MakeCurrent is really important step.
            For the NVHANDLE to succeed.
        */

        // Make GLX context current (required before NvFBC handle/session creation)
        // Use glXMakeContextCurrent (newer preferred way)
        glXMakeContextCurrent_ptr = (PFNGLXMAKECONTEXTCURRENTPROC)glXGetProcAddress((const GLubyte*)"glXMakeContextCurrent");
        Bool make_current_ok = False;
        // glXMakeContextCurrent_ptr = NULL;
        if (glXMakeContextCurrent_ptr) {
            printf("Creating pbuffer or window\n");
            #ifdef USE_PBUFFER
            make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
            #else
            make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, dummy_win, dummy_win, data.glx_ctx);
            #endif
        } else {
             fprintf(stdout, "GLX Capture Thread: glXMakeContextCurrent not found, trying glXMakeCurrent.\n");
             // Fallback to glXMakeCurrent if glXMakeContextCurrent failed/unavailable
             #ifdef USE_PBUFFER
             make_current_ok = glXMakeCurrent(data.glx_dpy, data.glx_pbuffer ? (GLXDrawable)data.glx_pbuffer : None, data.glx_ctx); // Direct call
             #else
             make_current_ok = glXMakeCurrent(data.glx_dpy, dummy_win ? (GLXDrawable)dummy_win : None, data.glx_ctx); 
             #endif
        }

        if (!make_current_ok) {
            fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current with Pbuffer/None.\n");
            break; // Exit setup block
        }

        fprintf(stdout, "GLX Capture Thread: GLX context is current.\n");
        printf("OPENGL Version: %s\n",glGetString(GL_VERSION));

        // --- Check if NvFBC API functions are loaded ---
        if (!nvFBC || !nvFBC->nvFBCCreateHandle) {
             fprintf(stderr, "GLX Capture Thread: NvFBC API functions not loaded correctly. Stopping.\n");
             break; // Exit setup block
        }
        fprintf(stdout, "GLX Capture Thread: NvFBC API function list seems valid.\n");

        // Create NvFBC handle
        pthread_mutex_lock(&data.session_state_mutex);
        
        NVFBC_CREATE_HANDLE_PARAMS handle_params;
        memset(&handle_params, 0, sizeof(handle_params));

        handle_params.dwVersion = NVFBC_CREATE_HANDLE_PARAMS_VER;

        
        #if 1
        handle_params.bExternallyManagedContext = NVFBC_TRUE; // We Handle
        handle_params.glxCtx = data.glx_ctx; // Pass the GLX context we created
        handle_params.glxFBConfig = chosen;
        #else
        handle_params.bExternallyManagedContext = NVFBC_FALSE; // Let NvFBC manage context state
        #endif

        NVFBCSTATUS ret = nvFBC->nvFBCCreateHandle(&data.nvfbc_session, &handle_params); // Use function pointer
         if (ret != NVFBC_SUCCESS) {
            #if 0
            fprintf(stderr, "GLX Capture Thread: nvFBCCreateHandle failed: %s\n", nvFBC->nvFBCGetLastErrorStr(NULL));
            pthread_mutex_unlock(&data.session_state_mutex);
            break;
            #endif

            /*
                Capture SDK can be only used on GRID, Tesla, or Quadro X2000+ (X = K/M/P/RTX etc) hardware
                products. Other configurations are not permitted under the end-user license agreement terms and
                conditions.
            */
            // Reverse engineering for interoperability
            const uint8_t enable_key[] = { 0xac, 0x10, 0xc9, 0x2e, 0xa5, 0xe6, 0x87, 0x4f, 0x8f, 0x4b, 0xf4, 0x61, 0xf8, 0x56, 0x27, 0xe9 };
            handle_params.privateData = enable_key;
            handle_params.privateDataSize = 16;

            ret = nvFBC->nvFBCCreateHandle(&data.nvfbc_session, &handle_params);
            if(ret != NVFBC_SUCCESS) {
                fprintf(stderr, "GLX Capture Thread: nvFBCCreateHandle failed: %s\n", nvFBC->nvFBCGetLastErrorStr(NULL));
                pthread_mutex_unlock(&data.session_state_mutex);
                break;
            }
         }
        pthread_mutex_unlock(&data.session_state_mutex);

        fprintf(stdout, "GLX Capture Thread: NVFBC handle created.\n");

        // If we reach here, setup was successful
        setup_success = true;

    } while (false); // End of setup block

    if (!setup_success) {
         data.stop_threads.store(true); // Signal other threads to stop on setup failure
    }


    // --- Main Capture Loop ---
    // Only run the loop if setup was successful
    if (setup_success) {
        // Get function pointer for glEGLImageTargetTexture2DOES if needed (it's declared in headers, but some drivers might not provide it directly)
        PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES_ptr = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
        if (!glEGLImageTargetTexture2DOES_ptr) {
             fprintf(stderr, "GLX Capture Thread: eglGetProcAddress failed to find glEGLImageTargetTexture2DOES. EGLImage binding will fail!\n");
             data.stop_threads.store(true); // Cannot proceed without this function
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

                #if 0
                    typedef enum _NVFBC_CAPTURE_TYPE
                    {
                        /*!
                         * Capture frames to a buffer in system memory.
                         */
                        NVFBC_CAPTURE_TO_SYS = 0,
                        /*!
                         * Capture frames to a CUDA device in video memory.
                         *
                         * Specifying this will dlopen() libcuda.so.1 and fail if not available.
                         */
                        NVFBC_CAPTURE_SHARED_CUDA = 1,
                        /*!
                         * Retired. Do not use.
                         */
                        /* NVFBC_CAPTURE_TO_HW_ENCODER, */
                        /*!
                         * Capture frames to an OpenGL buffer in video memory.
                         */
                        NVFBC_CAPTURE_TO_GL = 3,
                    } NVFBC_CAPTURE_TYPE;
                #endif
            
                // Create new session with hardcoded settings
                NVFBC_CREATE_CAPTURE_SESSION_PARAMS cap_params = {
                    .dwVersion = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER,
                    .eCaptureType = NVFBC_CAPTURE_TO_GL,
                    .eTrackingType = data.screen_id == -1 ? NVFBC_TRACKING_SCREEN : NVFBC_TRACKING_OUTPUT,
                    .dwOutputId = data.screen_id,
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
                    usleep(100000); // Sleep 100ms before retrying
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
                ret = nvFBC->nvFBCToGLSetUp(data.nvfbc_session, &data.togl_setup_params); // Use function pointer
                pthread_mutex_unlock(&data.session_state_mutex);

                if (ret != NVFBC_SUCCESS) {
                    fprintf(stderr, "GLX Capture Thread: nvFBCToGLSetUp failed during recreate: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
                    // Destroy the failed session
                    pthread_mutex_lock(&data.session_state_mutex);
                    NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
                    nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use function pointer
                    pthread_mutex_unlock(&data.session_state_mutex);
                    data.has_capture_session.store(false);
                    usleep(100000); // Sleep 100ms before retrying
                    continue; // Skip to next loop iteration
                }
                data.has_capture_session.store(true);
                fprintf(stdout, "GLX Capture Thread: NVFBC session recreated successfully.\n");
            } // end recreate_session check

            if (!data.has_capture_session.load()) {
                // Wait if session creation failed or is pending
                usleep(100000);
                continue; // Skip to next loop iteration
            }

            // Ensure GLX context is current for grab and EGLImage creation
            Bool make_current_ok = False;
            if (glXMakeContextCurrent_ptr) {
                 make_current_ok = glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
            } else {
                 make_current_ok = glXMakeCurrent(data.glx_dpy, data.glx_pbuffer ? (GLXDrawable)data.glx_pbuffer : None, data.glx_ctx); // Direct call
            }

            if (!make_current_ok) {
                fprintf(stderr, "GLX Capture Thread: Failed to make GLX context current for grab! Fatal.\n");
                data.stop_threads.store(true);
                continue; // Exit loop via stop_threads check
            }

            // Grab frame
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

            bool process_frame = false;
            EGLImageKHR img = EGL_NO_IMAGE_KHR; // Initialize image handle

            if (ret == NVFBC_SUCCESS && info.bIsNewFrame == NVFBC_TRUE) {
                 glxTex = data.togl_setup_params.dwTextures[grab_params.dwTextureIndex];
                 if (glxTex != 0) {
                      process_frame = true;
                      // Update dimensions atomically
                      data.frame_width.store(info.dwWidth);
                      data.frame_height.store(info.dwHeight);
                      //fprintf(stdout, "GLX Capture Thread: Grabbed new frame %ux%u, texture %u.\n", info.dwWidth, info.dwHeight, glxTex);
                 } else {
                     fprintf(stderr, "GLX Capture Thread: Grab succeeded but returned texture ID 0.\n");
                 }
            } else {
                 // Error other than timeout (timeout is expected with NOWAIT if no new frame)
                 fprintf(stderr, "GLX Capture Thread: nvFBCToGLGrabFrame failed: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
            }

            // Create EGLImage from the GLX texture if we have a new frame and the required function
            if (process_frame && glEGLImageTargetTexture2DOES_ptr) {
                const EGLint attrs[] = {
                    EGL_IMAGE_PRESERVED_KHR, EGL_TRUE,
                    EGL_NONE
                };
                
                /*
                Requires OpenGL 4.2, OpenGL ES 3.0, or ARB_texture_storage.

                Requires EGL 1.4 and either the EGL_KHR_image or EGL_KHR_image_base
                extensions.
                */
                // Use the thread-local EGL display and the GLX texture handle

                //KHR For EGL 1.4 and earlier
                // /data.shared_egl_dpy, //glx_thread_local_egl_dpy
                img = p_eglCreateImageKHR(
                    data.glx_thread_local_egl_dpy,
                    EGL_NO_CONTEXT,//eglGetCurrentContext(), // Use the current GLX context bound as EGL context
                    EGL_GL_TEXTURE_2D, // Assuming this target works for GLX textures
                    (EGLClientBuffer)(uintptr_t)glxTex,
                    attrs
                );
                
                if (img == EGL_NO_IMAGE_KHR) {
                     fprintf(stderr, "GLX Capture Thread: Failed to create EGLImage from GLX texture %u: %d\n", glxTex, eglGetError());
                     process_frame = false; // Don't indicate success if image creation failed
                } // else fprintf(stdout, "GLX Capture Thread: Created EGLImage %p from GLX texture %u.\n", img, glxTex);
            } else if (process_frame && !glEGLImageTargetTexture2DOES_ptr) {
                 process_frame = false; // Cannot process frame if EGLImage binding function is missing
            }


            // Push the image handle (EGL_NO_IMAGE_KHR if creation failed or no new frame)
            data.imageQueue.push(img);
            //fprintf(stdout, "GLX Capture Thread: Pushed image handle %p to queue.\n", img);

            // Unbind GLX context after operations in this iteration
            if (glXMakeContextCurrent_ptr) {
                glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL);
            } else {
                 glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
            }
            //fprintf(stdout, "GLX Capture Thread: Unbound GLX context.\n");


            // Sleep for FPS control or on error/timeout
            if (process_frame) {
                 if (data.fps > 0) {
                     usleep(1000000 / data.fps); // sleep microseconds
                } else {
                     usleep(1000); // short sleep
                }
            } else {
                 // Sleep on other errors or if setup failed during loop
                 usleep(100000); // Sleep 100ms
            }

        } // End of capture loop
    }


    // --- Cleanup ---
    fprintf(stdout, "GLX Capture Thread: Cleaning up.\n");

    // Ensure NvFBC API functions are loaded before attempting to use them for cleanup
    // If NvFBC handle creation failed, nvFBC pointer might be partially invalid.
    if (nvFBC && nvFBC->nvFBCDestroyCaptureSession && nvFBC->nvFBCDestroyHandle && nvFBC->nvFBCGetLastErrorStr) {
        // Destroy NVFBC session and handle
        pthread_mutex_lock(&data.session_state_mutex);
        if (data.has_capture_session.load()) {
            NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
            NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyCaptureSession(data.nvfbc_session, &destroy_cap_params); // Use function pointer
            if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture Thread: nvFBCDestroyCaptureSession failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer
            data.has_capture_session.store(false);
        }
        // Context must be current if bExternallyManagedContext=FALSE during DestroyHandle
        PFNGLXMAKECONTEXTCURRENTPROC glXMakeContextCurrent_ptr = (PFNGLXMAKECONTEXTCURRENTPROC)glXGetProcAddress((const GLubyte*)"glXMakeContextCurrent");
        if (data.nvfbc_session != -1) {
             if (glXMakeContextCurrent_ptr) {
                  glXMakeContextCurrent_ptr(data.glx_dpy, data.glx_pbuffer, data.glx_pbuffer, data.glx_ctx);
             } else {
                  glXMakeCurrent(data.glx_dpy, data.glx_pbuffer ? (GLXDrawable)data.glx_pbuffer : None, data.glx_ctx); // Direct call
             }

            NVFBC_DESTROY_HANDLE_PARAMS destroy_params = {.dwVersion = NVFBC_DESTROY_HANDLE_PARAMS_VER};


            NVFBCSTATUS dest_ret = nvFBC->nvFBCDestroyHandle(data.nvfbc_session, &destroy_params); // Use function pointer
            if (dest_ret != NVFBC_SUCCESS) fprintf(stderr, "GLX Capture Thread: nvFBCDestroyHandle failed during cleanup: %s\n", nvFBC->nvFBCGetLastErrorStr(data.nvfbc_session)); // Use function pointer

            if (glXMakeContextCurrent_ptr) {
                 glXMakeContextCurrent_ptr(data.glx_dpy, None, None, NULL); // Release context
            } else {
                 glXMakeCurrent(data.glx_dpy, None, NULL); // Direct call
            }
            data.nvfbc_session = -1;
        }
        pthread_mutex_unlock(&data.session_state_mutex);
    } else {
         fprintf(stderr, "GLX Capture Thread: NvFBC API functions not fully available for cleanup.\n");
    }


    // Destroy GLX context and Pbuffer
    if (data.glx_ctx) {
        // Use function pointer if available, fallback to direct call
        // PFNGLXDESTROYCONTEXTPROC glXDestroyContext_ptr = (PFNGLXDESTROYCONTEXTPROC)glXGetProcAddress((const GLubyte*)"glXDestroyContext");
        // if (glXDestroyContext_ptr) glXDestroyContext_ptr(data.glx_dpy, data.glx_ctx);
        // else glXDestroyContext(data.glx_dpy, data.glx_ctx); // Direct call
        glXDestroyContext(data.glx_dpy, data.glx_ctx); // Direct call
        data.glx_ctx = NULL;
    }
    if (data.glx_pbuffer) {
         // Use function pointer if available, fallback to direct call
         // PFNGLXDESTROYPBUFFERPROC glXDestroyPbuffer_ptr = (PFNGLXDESTROYPBUFFERPROC)glXGetProcAddress((const GLubyte*)"glXDestroyPbuffer");
         // if (glXDestroyPbuffer_ptr) glXDestroyPbuffer_ptr(data.glx_dpy, data.glx_pbuffer);
         // else glXDestroyPbuffer(data.glx_dpy, data.glx_pbuffer); // Direct call
        glXDestroyPbuffer(data.glx_dpy, data.glx_pbuffer); // Direct call
        data.glx_pbuffer = None;
    }

    // Terminate thread-local EGL
     if (data.glx_thread_local_egl_dpy != EGL_NO_DISPLAY) {
         eglTerminate(data.glx_thread_local_egl_dpy); // Direct call
         data.glx_thread_local_egl_dpy = EGL_NO_DISPLAY;
     }

    // Close X11 display connection
    if (data.glx_dpy) {
        XCloseDisplay(data.glx_dpy); // Direct call
        data.glx_dpy = NULL;
    }

    // Signal the EGL processor thread to stop
    data.imageQueue.stop_queue();

    fprintf(stdout, "GLX Capture Thread: Exited.\n");
    return NULL;
}

// --- EGL Processor Thread Function ---
static void *egl_processor_thread(void *param)
{
    fprintf(stdout, "EGL Processor Thread: Started.\n");

    // This thread uses the SHARED EGL Display initialized by the main thread.
    EGLDisplay dpy = data.shared_egl_dpy; // This should be valid from main()

    if (dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "EGL Processor Thread: Shared EGL Display is invalid! Fatal.\n");
        data.stop_threads.store(true); // Signal GLX thread too
        // Cleanup happens naturally as the function exits
    } else {
        EGLImageKHR image_for_main = EGL_NO_IMAGE_KHR; // The image handle currently visible to the main thread

        while (!data.stop_threads.load() || data.imageQueue.size() > 0) { // Continue processing queue until empty if stopping

            // Wait for a new EGLImage from the capture thread
            EGLImageKHR new_image = data.imageQueue.pop(); // Blocking pop, respects stop flag

            if (new_image == EGL_NO_IMAGE_KHR) { // Stop signal received or queue empty on stop
                if (data.stop_threads.load() && data.imageQueue.size() == 0) {
                    fprintf(stdout, "EGL Processor Thread: Stop signal received and queue is empty.\n");
                    break; // Truly stopping the loop
                } else {
                     // Received EGL_NO_IMAGE_KHR unexpectedly from the queue (e.g., creation failed in capture thread)
                     fprintf(stderr, "EGL Processor Thread: Received EGL_NO_IMAGE_KHR unexpectedly.\n");
                     continue; // Skip processing this invalid image
                }
            }
            // else fprintf(stdout, "EGL Processor Thread: Popped a new EGLImage handle %p.\n", new_image);


            // Manage the lifecycle of the previous image and swap in the new one
            pthread_mutex_lock(&data.frame_state_mutex); // Protect shared frame state

            // Destroy the old image *before* making the new one available
            // Only destroy if it was a valid handle and we have a valid EGL display
            if (image_for_main != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY) {
                if (p_eglDestroyImageKHR(dpy, image_for_main) != EGL_TRUE) { // Direct call (EGL API)
                    fprintf(stderr, "EGL Processor Thread: Failed to destroy old EGLImage %p: %d\n", image_for_main, eglGetError()); // Direct call
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
        fprintf(stdout, "EGL Processor Thread: Cleaning up.\n");

        // Destroy any remaining image handle we were managing
        pthread_mutex_lock(&data.frame_state_mutex);
        if (image_for_main != EGL_NO_IMAGE_KHR && dpy != EGL_NO_DISPLAY) {
            if (p_eglDestroyImageKHR(dpy, image_for_main) != EGL_TRUE) { // Direct call (EGL API)
                fprintf(stderr, "EGL Processor Thread: Failed to destroy final EGLImage during cleanup: %d\n", eglGetError()); // Direct call
            } // else fprintf(stdout, "EGL Processor Thread: Destroyed final EGLImage %p.\n", image_for_main);
        }
        data.current_render_image = EGL_NO_IMAGE_KHR; // Reset state
        data.new_image_available.store(false);
        pthread_mutex_unlock(&data.frame_state_mutex);

        // Clear any images left in the queue (should be empty if GLX thread finished correctly, but defensive)
        data.imageQueue.clear(dpy);
    }


    fprintf(stdout, "EGL Processor Thread: Exited.\n");
    return NULL;
}


// --- Main Application Functions ---

// Placeholder render function (draws a simple textured quad)
// Assumes main_egl_ctx is current.
void render_texture(GLuint tex, uint32_t width, uint32_t height) {
    if (tex == 0 || width == 0 || height == 0) return;

    // Minimal VAO/VBO for a quad - simplifies drawing
    static GLuint vao = 0;
    static GLuint vbo = 0;
    // Using GL calls directly as they are linked via libGL/libGLX
    if (vao == 0) {
        glGenVertexArrays(1, &vao); // Direct call
        glGenBuffers(1, &vbo); // Direct call

        float vertices[] = {
            // Positions        // Texture Coords (adjust if Y needs flipping)
            -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // Corrected Y for typical desktop capture (origin top-left)
             1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 0.0f
        };

        glBindVertexArray(vao); // Direct call
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // Direct call
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // Direct call

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0); // Direct call
        glEnableVertexAttribArray(0); // Direct call
        // Texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))); // Direct call
        glEnableVertexAttribArray(1); // Direct call

        glBindBuffer(GL_ARRAY_BUFFER, 0); // Direct call
        glBindVertexArray(0); // Direct call
    }

    // Minimal shader - you need to compile and use an actual shader program here.
    // This example only shows binding the texture. A real render needs shaders.
    // glUseProgram(your_shader_program); // Direct call
    // glUniform1i(texture_sampler_uniform_location, 0); // Direct call (if using texture unit 0)


    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Direct call
    glClear(GL_COLOR_BUFFER_BIT); // Direct call

    // Activate texture unit (e.g., 0) and bind the texture
    glActiveTexture(GL_TEXTURE0); // Direct call
    glBindTexture(GL_TEXTURE_2D, tex); // Direct call

    // Using VAO/VBO approach which requires shaders
    glBindVertexArray(vao); // Direct call
    glDrawArrays(GL_QUADS, 0, 4); // Draw the quad // Direct call
    glBindVertexArray(0); // Direct call

    glBindTexture(GL_TEXTURE_2D, 0); // Unbind // Direct call
    // glUseProgram(0); // Unbind shader program (if used) // Direct call


    GLenum err = glGetError(); // Direct call
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "GL error during render_texture: %x\n", err);
    }

     eglSwapBuffers(data.shared_egl_dpy, eglGetCurrentSurface(EGL_DRAW)); // Direct call
}


// Helper to check GL extensions
bool check_gl_ext(const char *name) {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) { // Direct call (EGL API)
        fprintf(stderr, "Cannot check GL extensions, no EGL context current.\n");
        return false;
    }

    GLint ext_count = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &ext_count); // Direct call (GL API)
    if (ext_count > 0) {
        for (GLint i = 0; i < ext_count; ++i) {
            const GLubyte *ext = glGetStringi(GL_EXTENSIONS, i); // Direct call (GL API)
            if (!ext) {
                fprintf(stderr, "Could not get OpenGL extension #%i\n", i);
            } else if (!strcmp((const char *)ext, name)) {
                return true;
            }
        }
    } else { // Fallback for older GL contexts or GLES
        const char *extensions = (const char *)glGetString(GL_EXTENSIONS); // Direct call (GL API)
        if (extensions) {
            return strstr(extensions, name) != NULL; // Simplified check
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
    const char *extensions = eglQueryString(dpy, EGL_EXTENSIONS); // Direct call (EGL API)
    if (!extensions) {
        fprintf(stderr, "Could not get EGL extensions string\n");
        return false;
    }
    // printf("EXTENSIONS:\n%s\n",extensions);
    return strstr(extensions, name) != NULL; // Simplified check
}


int main() {
    fprintf(stdout, "Minimal NvFBC EGL/GLX Test (Linux)\n");

    // --- 0. Initialize shared data mutexes ---
    // These are initialized unconditionally, so they are destroyed unconditionally at the end.
    pthread_mutex_init(&data.session_state_mutex, NULL);
    pthread_mutex_init(&data.frame_state_mutex, NULL);

    bool main_setup_success = false; // Flag to track if main thread setup completed successfully

    // Use separate flags for each resource created during setup
    bool nvfbc_lib_loaded = false;
    bool nvfbc_instance_created = false; // Indicates data.nvFBC_api is populated
    bool main_x_dpy_opened = false;
    bool main_x_win_created = false;
    bool shared_egl_dpy_initialized = false;
    bool main_egl_config_chosen = false; // Though not strictly a resource to destroy
    bool main_egl_ctx_created = false;
    bool main_egl_surface_created = false;
    pthread_t glx_thread_handle = 0; // Use 0 to indicate not created
    pthread_t egl_processor_thread_handle = 0; // Use 0 to indicate not created


    do { // Use a do-while(false) loop to allow using break for early exit during setup
        // --- Load NvFBC Library Dynamically ---
        fprintf(stdout, "Loading NvFBC library...\n");
        data.nvfbc_lib_handle = os_dlopen("libnvidia-fbc.so.1");
        if (!data.nvfbc_lib_handle) {
            fprintf(stderr, "Error: Failed to load libnvidia-fbc.so.1. Please ensure the NVIDIA driver is installed and the library is in the linker path.\n");
            break; // Exit setup block
        }
        nvfbc_lib_loaded = true;
        fprintf(stdout, "NvFBC library loaded.\n");

        // --- Get NvFBCCreateInstance function pointer ---
        PFNNVFBCCreateInstance NvFBCCreateInstance_ptr = (PFNNVFBCCreateInstance)os_dlsym(data.nvfbc_lib_handle, "NvFBCCreateInstance");

        if (!NvFBCCreateInstance_ptr) {
            fprintf(stderr, "Error: Failed to get NvFBCCreateInstance symbol from the loaded library. Symbol not found or mismatch.\n");
            break; // Exit setup block
        }
        fprintf(stdout, "NvFBCCreateInstance symbol found.\n");

        // Load required EGL extension functions
        p_eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImage");
        p_eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImage");
        p_glEGLImageTargetTexture2DOES = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");

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
             fprintf(stderr, "Error: NvFBCCreateInstance succeeded, but some required function pointers are NULL. API mismatch?\n");
             // Consider this a fatal setup error as NvFBC won't work
             break; // Exit setup block
        }
        fprintf(stdout, "NvFBC API function pointers verified.\n");


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
        #if 0
        XMapWindow(data.main_x_dpy, data.main_x_win); // Direct call
        XFlush(data.main_x_dpy); // Direct call
        #endif

        #if 1
        // Get EGL display for the main X11 display
        data.shared_egl_dpy = eglGetDisplay((EGLNativeDisplayType)data.main_x_dpy); // Direct call
        if (data.shared_egl_dpy == EGL_NO_DISPLAY) {
            fprintf(stderr, "Error: Failed to get main EGL Display.\n");
            break; // Exit setup block
        }

        EGLint major, minor;
        if (!eglInitialize(data.shared_egl_dpy, &major, &minor)) { // Direct call
            fprintf(stderr, "Error: Failed to initialize main EGL Display: %d\n", eglGetError()); // Direct call
            break; // Exit setup block
        }
        shared_egl_dpy_initialized = true;
        fprintf(stdout, "Main thread EGL Display initialized (version %d.%d).\n", major, minor);

        // Check required EGL extensions on the shared display
        if (!check_egl_ext(data.shared_egl_dpy, "EGL_KHR_image_base")) {
            fprintf(stderr, "Error: Required EGL extension 'EGL_KHR_image_base' not supported.\n"); break;
        }
        if (!check_egl_ext(data.shared_egl_dpy, "EGL_KHR_gl_texture_2D_image")) {
             fprintf(stderr, "Error: Required EGL extension 'EGL_KHR_gl_texture_2d_image' not supported.\n"); break;
        }
         fprintf(stdout, "Required EGL extensions found.\n");


        // Choose an EGL config for the main context
        const EGLint config_attrs[] = {
            EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, // Request OpenGL support EGL_OPENGL_ES_BIT
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
        main_egl_config_chosen = true; // Flag indicates config pointer is potentially valid
        fprintf(stdout, "Main thread EGL config chosen.\n");

        // Create main thread EGL context
        static const EGLint context_attribs[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE }; // Request OpenGL 3.3+
        data.main_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, EGL_NO_CONTEXT, context_attribs); // Direct call
         if (data.main_egl_ctx == EGL_NO_CONTEXT) {
              fprintf(stdout, "Error: Failed to create main EGL context (3.3+), trying GLES 2.\n");
               static const EGLint context_attribs_gles2[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE }; // Try GLES 2
               data.main_egl_ctx = eglCreateContext(data.shared_egl_dpy, data.shared_egl_config, EGL_NO_CONTEXT, context_attribs_gles2); // Direct call
               if (data.main_egl_ctx == EGL_NO_CONTEXT) {
                    fprintf(stderr, "Error: Failed to create any main EGL context: %d.\n", eglGetError()); // Direct call
                    break; // Exit setup block
               }
         }
        main_egl_ctx_created = true;
        fprintf(stdout, "Main thread EGL context created.\n");

        // Create EGL surface for the main window
        data.main_egl_surface = eglCreateWindowSurface(data.shared_egl_dpy, data.shared_egl_config, (EGLNativeWindowType)data.main_x_win, NULL); // Direct call
         if (data.main_egl_surface == EGL_NO_SURFACE) {
             fprintf(stderr, "Error: Failed to create main EGL window surface: %d.\n", eglGetError()); // Direct call
             break; // Exit setup block
         }
        main_egl_surface_created = true;
        fprintf(stdout, "Main thread EGL surface created.\n");

        // Make main context current (needed for GL calls like texture creation and extension checks)
        if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) { // Direct call
            fprintf(stderr, "Error: Failed to make main EGL context current: %d\n", eglGetError()); // Direct call
            break; // Exit setup block
        }
        fprintf(stdout, "Main thread EGL context is current.\n");

        // Check required GL extension (GL_OES_EGL_image)
        if (!check_gl_ext("GL_OES_EGL_image")) {
             fprintf(stdout, "Note: GL_OES_EGL_image extension check failed. Binding EGLImage to GL texture relies on implementation support.\n");
        } else {
            fprintf(stdout, "GL_OES_EGL_image extension seems available.\n");
        }

        #endif
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
            data.stop_threads.store(true); // Signal GLX thread to stop
            pthread_join(glx_thread_handle, NULL); // Wait for GLX thread cleanup before main cleanup
            glx_thread_handle = 0; // Mark as joined
            break; // Exit setup block
        }
        fprintf(stdout, "EGL Processor Thread created.\n");

        // If we reached here, all main setup was successful
        main_setup_success = true;

    } while (false); // End of setup block


    // --- 3. Main Loop (Simulate OBS Render Loop) ---
    if (main_setup_success) {
        fprintf(stdout, "Entering main render loop. Close window to stop.\n");

        // Atom for window close message
        Atom wm_delete_window;
        wm_delete_window = XInternAtom(data.main_x_dpy, "WM_DELETE_WINDOW", False); // Direct call
        XSetWMProtocols(data.main_x_dpy, data.main_x_win, &wm_delete_window, 1); // Direct call

        // Minimal render loop (checks for updates and binds texture)
        while (!data.stop_threads.load()) {
            // Poll for new image availability
            bool new_image_ready = data.new_image_available.load();

            if (new_image_ready) {
                // fprintf(stdout, "Main Thread: New image available signal received.\n");

                pthread_mutex_lock(&data.frame_state_mutex); // Lock to read shared image handle and dimensions

                // Reset signal immediately after acquiring mutex
                data.new_image_available.store(false);

                EGLImageKHR image_to_bind = data.current_render_image;
                uint32_t width = data.frame_width.load();
                uint32_t height = data.frame_height.load();

                pthread_mutex_unlock(&data.frame_state_mutex); // Unlock after reading

                if (image_to_bind != EGL_NO_IMAGE_KHR) {
                    // fprintf(stdout, "Main Thread: Processing new EGLImage %p (%ux%u).\n", image_to_bind, width, height);

                    // Ensure context is current for GL calls before creating/deleting textures
                     if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) { // Direct call
                         fprintf(stderr, "Main Thread: Failed to make main EGL context current for texture operations: %d\n", eglGetError()); // Direct call
                         // Cannot proceed with this image if context fails
                     } else {
                        // Destroy the old main texture if it exists or dimensions changed
                        if (data.main_render_tex != 0 && (data.frame_width.load() != width || data.frame_height.load() != height)) {
                            glDeleteTextures(1, &data.main_render_tex); // Direct call
                            data.main_render_tex = 0;
                            fprintf(stdout, "Main Thread: Destroyed old render texture due to size change.\n");
                        }

                         // Create new main texture if needed and dimensions are valid
                        if (data.main_render_tex == 0 && width > 0 && height > 0) {
                             glGenTextures(1, &data.main_render_tex); // Direct call
                             if (data.main_render_tex == 0) {
                                 fprintf(stderr, "Main Thread: Failed to create main render texture.\n");
                             } else {
                                  glBindTexture(GL_TEXTURE_2D, data.main_render_tex); // Direct call
                                  // Allocate storage for the texture
                                  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); // Direct call
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Direct call
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Direct call
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Direct call
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Direct call
                                  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0); // No mipmaps needed // Direct call
                                  glBindTexture(GL_TEXTURE_2D, 0); // Direct call
                                  fprintf(stdout, "Main Thread: Created new render texture %u (%ux%u).\n", data.main_render_tex, width, height);
                             }
                        }

                        // If we have a valid texture and image, bind the image to the texture
                        if (data.main_render_tex != 0) {
                            // glEGLImageTargetTexture2DOES is typically provided by the driver's GL library
                            // We need to get its address if not linked directly by the header/lib combination
                            PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES_ptr = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
                            if (!glEGLImageTargetTexture2DOES_ptr) {
                                fprintf(stderr, "Main Thread: eglGetProcAddress failed to find glEGLImageTargetTexture2DOES. Cannot bind image.\n");
                                // Binding failed, texture might be invalid now. Destroy it.
                                glDeleteTextures(1, &data.main_render_tex); // Direct call
                                data.main_render_tex = 0;
                            } else {
                                /*
                                    EGLImageTargetTexture2DOES glTexImage2D except no upload.
                                */
                                glBindTexture(GL_TEXTURE_2D, data.main_render_tex); // Direct call
                                glEGLImageTargetTexture2DOES_ptr(GL_TEXTURE_2D, image_to_bind); // Call via pointer
                                glBindTexture(GL_TEXTURE_2D, 0); // Direct call

                                GLenum glerr = glGetError(); // Direct call
                                if (glerr != GL_NO_ERROR) {
                                     fprintf(stderr, "Main Thread: glEGLImageTargetTexture2DOES GL error: %x\n", glerr);
                                     // Binding failed, texture might be invalid now. Destroy it.
                                     glDeleteTextures(1, &data.main_render_tex); // Direct call
                                     data.main_render_tex = 0;
                                } // else fprintf(stdout, "Main Thread: Bound EGLImage %p to texture %u.\n", image_to_bind, data.main_render_tex);
                            }
                        }
                        // Context remains current after these ops
                     }
                } else {
                    // Received EGL_NO_IMAGE_KHR (but not the stop sentinel) - indicates a problem.
                    // Keep the last good frame, log a warning.
                    fprintf(stderr, "Main Thread: Received signal but image handle is EGL_NO_IMAGE_KHR from processor.\n");
                }
            }

            // Minimal render (optional, only if main_render_tex is valid)
            // Ensure context is current before rendering
            if (main_egl_ctx_created && main_egl_surface_created) { // Check if context/surface were created
                 if (!eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) { // Direct call
                      fprintf(stderr, "Main Thread: Failed to make main EGL context current for rendering: %d\n", eglGetError()); // Direct call
                 } else {
                      if (data.main_render_tex != 0) {
                          render_texture(data.main_render_tex, data.frame_width.load(), data.frame_height.load()); // Draw the texture
                      } else {
                          // Clear window if no texture to show
                          glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background // Direct call
                          glClear(GL_COLOR_BUFFER_BIT); // Direct call
                          eglSwapBuffers(data.shared_egl_dpy, data.main_egl_surface); // Direct call
                      }
                      // Context remains current for the next loop iteration
                 }
            }


            // Handle X events for window close (simple poll)
            XEvent event;
            while (main_x_dpy_opened && XPending(data.main_x_dpy)) { // Direct call
                XNextEvent(data.main_x_dpy, &event); // Direct call
                if (event.type == ClientMessage) {
                     // Check for window manager delete message
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


    // Destroy main thread GL texture IF EGL context was created
    if (main_egl_ctx_created && main_egl_surface_created) {
         // Ensure context is current for GL calls
         if (eglMakeCurrent(data.shared_egl_dpy, data.main_egl_surface, data.main_egl_surface, data.main_egl_ctx)) { // Direct call
             if (data.main_render_tex != 0) {
                 glDeleteTextures(1, &data.main_render_tex); // Direct call
                 data.main_render_tex = 0;
                 fprintf(stdout, "Main thread render texture destroyed.\n");
             }
             // Unmake context current
             eglMakeCurrent(data.shared_egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // Direct call
         } else {
             fprintf(stderr, "Failed to make main EGL context current for texture cleanup. Texture might leak.\n"); // Direct call
         }
    }


    // Destroy EGL surface IF created
    if (main_egl_surface_created) {
        eglDestroySurface(data.shared_egl_dpy, data.main_egl_surface); // Direct call
        data.main_egl_surface = EGL_NO_SURFACE;
    }

    // Destroy EGL context IF created
    if (main_egl_ctx_created) {
        eglDestroyContext(data.shared_egl_dpy, data.main_egl_ctx); // Direct call
        data.main_egl_ctx = EGL_NO_CONTEXT;
    }

    // Terminate EGL display IF initialized
    if (shared_egl_dpy_initialized) {
        eglTerminate(data.shared_egl_dpy); // Direct call
        data.shared_egl_dpy = EGL_NO_DISPLAY;
    }

    // Close NvFBC Library Handle IF loaded
    if (nvfbc_lib_loaded) {
        fprintf(stdout, "Closing NvFBC library handle.\n");
        os_dlclose(data.nvfbc_lib_handle);
        data.nvfbc_lib_handle = NULL;
    }

    // Destroy main X11 window IF created
     if (main_x_win_created && data.main_x_dpy) {
          XDestroyWindow(data.main_x_dpy, data.main_x_win); // Direct call
          data.main_x_win = 0;
     }

    // Close X11 display IF opened
    if (main_x_dpy_opened) {
        XCloseDisplay(data.main_x_dpy); // Direct call
        data.main_x_dpy = NULL;
    }

    // Destroy mutexes (initialized unconditionally)
    pthread_mutex_destroy(&data.session_state_mutex);
    pthread_mutex_destroy(&data.frame_state_mutex);


    fprintf(stdout, "Cleanup complete. Exiting.\n");
    return 0;
}