#include <X11/Xlib.h>
#include <GL/glx.h>
#include <GL/gl.h>
#include "NvFBC.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include <stdbool.h> // For bool
#include <unistd.h> // For usleep

#include <dlfcn.h>  // Required for dlopen, dlsym, dlclose, dlerror

// --- NVFBC Dynamic Loading Variables ---
// Type for the main NvFBCCreateInstance function
typedef NVFBCSTATUS (*PFNNVFBCCreateInstance)(NVFBC_API_FUNCTION_LIST* pFunctions);

// Global pointer to the NvFBCCreateInstance function
PFNNVFBCCreateInstance pNvFBCCreateInstance = NULL;

// Global struct to hold all NVFBC function pointers populated by NvFBCCreateInstance
NVFBC_API_FUNCTION_LIST nvFBC_api_functions;

// NVFBC dynamic library handle
void *nvfbc_lib_handle = NULL;

// NVFBC API handle (from nvFBCnvFBCCreateHandle)
NVFBC_SESSION_HANDLE nvfbcApiHandle = 0;


// ToGL Setup parameters (needed to access allocated textures)
NVFBC_TOGL_SETUP_PARAMS togl_setup_params;

// X11/GLX variables
XEvent event;
Display *display = NULL;
Window window = 0;
GLXContext glContext = NULL;
GLXFBConfig chosen_fb_config = 0; // Need FBConfig for nvFBCnvFBCCreateHandle
XVisualInfo *visual_info = NULL;
Colormap colormap = 0;

// Flags to track initialization status for cleanup
bool nvfbc_lib_loaded = false;
bool nvfbc_api_created = false;
bool nvfbc_handle_created = false;
bool nvfbc_capture_session_created = false;
bool glx_initialized = false;


// Function to cleanup NVFBC and GLX resources
void cleanup_resources() {

    // 1. Destroy NVFBC Capture Session
    // Context might need to be current for DestroySession depending on flags,
    // but usually DestroySession does not require it. nvFBCDestroyHandle might.
    // Let's try without making current first for simplicity.
    if (nvfbc_capture_session_created && nvfbcApiHandle && nvFBC_api_functions.nvFBCDestroyCaptureSession) {
        NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroy_cap_params = {.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER};
        NVFBCSTATUS ret = nvFBC_api_functions.nvFBCDestroyCaptureSession(nvfbcApiHandle, &destroy_cap_params);
        if (ret != NVFBC_SUCCESS) {
            fprintf(stderr, "NVFBC Cleanup: nvFBCnvFBCDestroyCaptureSession failed: %s\n",
                    nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
        } else {
            fprintf(stdout, "NVFBC Cleanup: Capture session destroyed.\n");
        }
        nvfbcApiHandle = 0;
        nvfbc_capture_session_created = false;
    }

    // 2. Destroy NVFBC API Handle
    // Context MUST be current for nvFBCDestroyHandle if bExternallyManagedContext was TRUE
    if (nvfbc_handle_created && nvfbcApiHandle && nvFBC_api_functions.nvFBCDestroyHandle) {
        // Ensure GLX context is current for this call
        if (glContext && display && chosen_fb_config) { // Check if GLX was successfully initialized
            // Need FBConfig to make context current with modern GLX functions
            // If glXChooseVisual was used, chosen_fb_config might be 0. Fallback to legacy.
            GLXDrawable drawable = window; // Use the main window as the drawable
             if (!glXMakeCurrent(display, drawable, glContext)) {
                 fprintf(stderr, "NVFBC Cleanup: Failed to make GLX context current for nvFBCnvFBCDestroyHandle!\n");
             }
        } else {
             fprintf(stderr, "NVFBC Cleanup: GLX context not available for nvFBCnvFBCDestroyHandle. Skipping making current.\n");
        }

        NVFBC_DESTROY_HANDLE_PARAMS destroy_params = {.dwVersion = NVFBC_DESTROY_HANDLE_PARAMS_VER};
        NVFBCSTATUS ret = nvFBC_api_functions.nvFBCDestroyHandle(nvfbcApiHandle, &destroy_params);

         // Unmake GLX context current after the NVFBC call
         if (glContext && display) {
             glXMakeCurrent(display, None, NULL);
         }

        if (ret != NVFBC_SUCCESS) {
             fprintf(stderr, "NVFBC Cleanup: nvFBCnvFBCDestroyHandle failed: %s\n",
                     nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
        } else {
            fprintf(stdout, "NVFBC Cleanup: NVFBC API handle destroyed.\n");
        }
        nvfbcApiHandle = 0;
        nvfbc_handle_created = false;
    }


    // 3. Destroy GLX/X11 resources
    if (glx_initialized) {
        glXMakeCurrent(display, None, NULL); // Release context
        if (glContext) {
            glXDestroyContext(display, glContext);
            glContext = NULL;
            fprintf(stdout, "GLX Cleanup: GLX context destroyed.\n");
        }
        if (window) {
            XDestroyWindow(display, window);
            window = 0;
            fprintf(stdout, "GLX Cleanup: X11 window destroyed.\n");
        }
        if (colormap) {
            XFreeColormap(display, colormap);
            colormap = 0;
            fprintf(stdout, "GLX Cleanup: X11 colormap freed.\n");
        }
        if (visual_info) { // If using glXChooseVisual, free this
            XFree(visual_info);
            visual_info = NULL;
            fprintf(stdout, "GLX Cleanup: XVisualInfo freed.\n");
        }
         // chosen_fb_config doesn't need explicit freeing
        if (display) {
            XCloseDisplay(display);
            display = NULL;
            fprintf(stdout, "GLX Cleanup: X11 display closed.\n");
        }
        glx_initialized = false;
    }


    // 4. Unload NVFBC library
    if (nvfbc_lib_loaded && nvfbc_lib_handle) {
        dlclose(nvfbc_lib_handle);
        nvfbc_lib_handle = NULL;
        nvfbc_lib_loaded = false;
        fprintf(stdout, "NVFBC Cleanup: NVFBC library unloaded.\n");

        // Reset function pointers after unloading
        pNvFBCCreateInstance = NULL;
        memset(&nvFBC_api_functions, 0, sizeof(nvFBC_api_functions));
        nvfbc_api_created = false;
    }
}

int main(int argc, char *argv[]) {

    Atom wmDeleteMessage;
    NVFBC_GET_STATUS_PARAMS statusParams;
    NVFBCSTATUS status;
    // Use do-while(false) for easy exit on setup failure
    do {
        // Register cleanup function to run on exit() or return from main
        atexit(cleanup_resources);

        // --- Dynamic NVFBC Loading and Instance Creation ---
        const char* nvfbc_lib_path = "libnvidia-fbc.so.1";

        fprintf(stdout, "Setup: Loading NVFBC library '%s'...\n", nvfbc_lib_path);
        nvfbc_lib_handle = dlopen(nvfbc_lib_path, RTLD_LAZY | RTLD_LOCAL);
        if (!nvfbc_lib_handle) {
            fprintf(stderr, "Setup Error: Failed to load NVFBC library '%s': %s\n", nvfbc_lib_path, dlerror());
            break; // Exit setup block
        }
        nvfbc_lib_loaded = true;
        fprintf(stdout, "Setup: NVFBC library loaded.\n");

        // --- Get NvFBCCreateInstance function pointer ---
        pNvFBCCreateInstance = (PFNNVFBCCreateInstance)dlsym(nvfbc_lib_handle, "NvFBCCreateInstance");
        if (!pNvFBCCreateInstance) {
            fprintf(stderr, "Setup Error: Failed to get NvFBCCreateInstance symbol from the loaded library: %s\n", dlerror());
            break; // Exit setup block
        }
        fprintf(stdout, "Setup: NvFBCCreateInstance symbol found.\n");

        // --- Call NvFBCCreateInstance via pointer to populate the struct ---
        memset(&nvFBC_api_functions, 0, sizeof(nvFBC_api_functions)); // Zero out the struct
        nvFBC_api_functions.dwVersion = NVFBC_VERSION; // Set the API version expected

        status = pNvFBCCreateInstance(&nvFBC_api_functions); // Call the loaded function

        if (status != NVFBC_SUCCESS) {
            fprintf(stderr, "Setup Error: NvFBCCreateInstance failed. Status: %d.\n", status);
             // We can't get a detailed error string here as the nvFBCGetLastErrorStr pointer might not be valid yet
            break; // Exit setup block
        }
        nvfbc_api_created = true;
        fprintf(stdout, "Setup: NvFBC instance created and API functions populated.\n");

        fprintf(stdout, "Setup: NvFBC API function pointers verified.\n");


        // --- X11 and GLX Initialization ---
        fprintf(stdout, "Setup: Initializing X11 and GLX...\n");
        display = XOpenDisplay(NULL);
        if (!display) {
            fprintf(stderr, "Setup Error: Cannot connect to X server\n");
            break; // Exit setup block
        }
        int screen = DefaultScreen(display);

        // 1. Choose GLX FBConfig (Required for nvFBCnvFBCCreateHandle with bExternallyManagedContext)
        // Choose a suitable FBConfig for Pbuffer or Window, focusing on RGBA and double buffering
        // Using GLX_WINDOW_BIT here as we're drawing to a window, not Pbuffer for minimal example
        /*
        GLint fb_attribs[] = {
            GLX_X_RENDERABLE, True,
            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT, // Use Window bit
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
            GLX_DEPTH_SIZE, 24, // Optional depth buffer
            GLX_DOUBLEBUFFER, True,
            None
        };*/
        GLint fb_attribs[] = {
            GLX_RENDER_TYPE, GLX_RGBA_BIT,
            GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
            // TODO:
            //GLX_BIND_TO_TEXTURE_RGBA_EXT, 1,
            //GLX_BIND_TO_TEXTURE_TARGETS_EXT, GLX_TEXTURE_2D_BIT_EXT,
            GLX_DOUBLEBUFFER, True,
            GLX_RED_SIZE, 8,
            GLX_GREEN_SIZE, 8,
            GLX_BLUE_SIZE, 8,
            GLX_ALPHA_SIZE, 0,
            GLX_DEPTH_SIZE, 0,
            None, None
        };
        int fb_count;
        GLXFBConfig *fb_configs = glXChooseFBConfig(display, screen, fb_attribs, &fb_count);
        if (fb_count == 0 || !fb_configs) {
            fprintf(stderr, "Setup Error: Failed to get GLXFBConfig for window.\n");
            if (fb_configs) XFree(fb_configs);
            break; // Exit setup block
        }
        chosen_fb_config = fb_configs[0]; // Select the first config
        XFree(fb_configs);
        fprintf(stdout, "Setup: Found %d GLX FBConfigs, using the first one.\n", fb_count);

        // 2. Get XVisualInfo from FBConfig (Needed for XCreateWindow)
        visual_info = glXGetVisualFromFBConfig(display, chosen_fb_config);
        if (!visual_info) {
             fprintf(stderr, "Setup Error: Failed to get XVisualInfo from GLX FBConfig.\n");
             break; // Exit setup block
        }

        // 3. Create Colormap and Window
        colormap = XCreateColormap(display, RootWindow(display, screen), visual_info->visual, AllocNone);
        XSetWindowAttributes swa;
        swa.colormap = colormap;
        // Add StructureNotifyMask to handle ConfigureNotify events (for resize)
        swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask | ClientMessage;

        // Get screen size for initial window dimensions
        
        memset(&statusParams, 0, sizeof(statusParams));
        statusParams.dwVersion = NVFBC_GET_STATUS_PARAMS_VER;
        status = nvFBC_api_functions.nvFBCGetStatus(nvfbcApiHandle, &statusParams); // Using the API handle for this call
         if (status != NVFBC_SUCCESS) {
             fprintf(stderr, "Setup Warning: Failed to get initial NVFBC status for screen size: %s. Using default 800x600.\n",
                     nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
             statusParams.screenSize.w = 800; // Default size if status fails
             statusParams.screenSize.h = 600; // Default size if status fails
         }

        window = XCreateWindow(display, RootWindow(display, screen),
                               0, 0, statusParams.screenSize.w, statusParams.screenSize.h, 0,
                               visual_info->depth, InputOutput, visual_info->visual,
                               CWColormap | CWEventMask, &swa);

        if (!window) {
             fprintf(stderr, "Setup Error: Failed to create X11 window.\n");
             break; // Exit setup block
        }

        // Set window title and protocol to handle close button
        XStoreName(display, window, "Minimal NVFBC Viewer");
        wmDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display, window, &wmDeleteMessage, 1);

        // 4. Create GLX Context (using ContextAttribs for modern OpenGL)
        // Try creating modern context first
        static int context_attribs[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 3, // Request OpenGL 3.3+
            GLX_CONTEXT_MINOR_VERSION_ARB, 1,
            GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
            None
        };

        #if 0
        // Need to get glXCreateContextAttribsARB via dlsym as it's an extension
        PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB_ptr = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB");

        if (glXCreateContextAttribsARB_ptr) {
             glContext = glXCreateContextAttribsARB_ptr(display, chosen_fb_config, NULL, True, context_attribs);
             if (!glContext) {
                 fprintf(stdout, "Setup: Modern GLX context creation failed (3.1 Core). Trying legacy context.\n");
             }
        } else {
             fprintf(stdout, "Setup: glXCreateContextAttribsARB not found. Trying legacy GLX context.\n");
        }
        #else
        glContext = glXCreateNewContext(display, chosen_fb_config, GLX_RGBA_TYPE, NULL, True);
        #endif

        // Fallback to legacy GLX context if modern creation failed or extension not found
        if (!glContext) {
            glContext = glXCreateContext(display, visual_info, NULL, True);
             if (!glContext) {
                 fprintf(stderr, "Setup Error: Failed to create any GLX context.\n");
                 break; // Exit setup block
             }
             fprintf(stdout, "Setup: Created legacy GLX context.\n");
        } else {
            fprintf(stdout, "Setup: Created modern GLX context.\n");
        }


        // 5. Make GLX Context Current
        if (!glXMakeCurrent(display, window, glContext)) { // Use the main window as drawable
             fprintf(stderr, "Setup Error: Failed to make GLX context current with window.\n");
             break; // Exit setup block
        }
        fprintf(stdout, "Setup: GLX context is current.\n");
        printf("Setup: GLX Context OpenGL Version: %s\n",glGetString(GL_VERSION));
        printf("Setup: GLX Context OpenGL Renderer: %s\n",glGetString(GL_RENDERER));

        // GLX Initialization successful
        glx_initialized = true;
        fprintf(stdout, "Setup: X11 and GLX initialized.\n");

        if (glXIsDirect(display, glContext)) {
            printf("Direct rendering is enabled.\n");
        } else {
            printf("Indirect rendering is used.\n");
        }

        // --- NVFBC Handle Creation ---
        // GLX context MUST be current before calling nvFBCnvFBCCreateHandle if bExternallyManagedContext is TRUE
        // It is already current from the previous step.
        fprintf(stdout, "Setup: Creating NVFBC API handle...\n");
        NVFBC_CREATE_HANDLE_PARAMS handle_params;
        memset(&handle_params, 0, sizeof(handle_params));
        handle_params.dwVersion = NVFBC_CREATE_HANDLE_PARAMS_VER;
        handle_params.bExternallyManagedContext = NVFBC_TRUE; // We manage contexts
        handle_params.glxCtx = glContext; // Pass the GLX context we created
        handle_params.glxFBConfig = chosen_fb_config; // Pass the FBConfig
        const uint8_t enable_key[] = { 0xac, 0x10, 0xc9, 0x2e, 0xa5, 0xe6, 0x87, 0x4f, 0x8f, 0x4b, 0xf4, 0x61, 0xf8, 0x56, 0x27, 0xe9 };
        handle_params.privateData = enable_key;
        handle_params.privateDataSize = 16;

        status = nvFBC_api_functions.nvFBCCreateHandle(&nvfbcApiHandle, &handle_params);

        if (status != NVFBC_SUCCESS) {
            fprintf(stderr, "Setup Error: nvFBCnvFBCCreateHandle failed: %s\n",
                    nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(0) : "Unknown Error (pre-handle)");
             // The nvFBCGetLastErrorStr might need a valid handle, use 0 if handle creation failed
            // Optional: try the workaround from your snippet if needed on your system
            // const uint8_t enable_key[] = { 0xac, 0x10, 0xc9, 0x2e, 0xa5, 0xe6, 0x87, 0x4f, 0x8f, 0x4b, 0xf4, 0x61, 0xf8, 0x56, 0x27, 0xe9 };
            // handle_params.privateData = enable_key;
            // handle_params.privateDataSize = 16;
            // status = nvFBC_api_functions.nvFBCCreateHandle(&nvfbcApiHandle, &handle_params);
            // if (status != NVFBC_SUCCESS) {
            //      fprintf(stderr, "Setup Error: nvFBCnvFBCCreateHandle failed after workaround: %s\n",
            //              nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(0) : "Unknown Error (pre-handle)");
            //      break;
            // }
        }

        if (status != NVFBC_SUCCESS) { // Check status again if workaround was attempted
            break; // Exit setup block
        }
        nvfbc_handle_created = true;
        fprintf(stdout, "Setup: NVFBC API handle created (%lu).\n", nvfbcApiHandle);

        // Unmake GLX context current after nvFBCnvFBCCreateHandle call (as seen in snippet)
        glXMakeCurrent(display, None, NULL);
        fprintf(stdout, "Setup: Unbound GLX context after nvFBCnvFBCCreateHandle.\n");


        // --- NVFBC Capture Session Creation ---
        // Re-make GLX context current for CaptureSession and Setup calls if needed
        if (!glXMakeCurrent(display, window, glContext)) {
             fprintf(stderr, "Setup Error: Failed to re-make GLX context current for NVFBC session setup.\n");
             break; // Exit setup block
        }
        fprintf(stdout, "Setup: Remade GLX context current.\n");

        //https://gitlab.com/fzwoch/obs-nvfbc/-/blob/master/NvFBC.h
        /*
            Direct capture is possible under the following conditions:
                 * - Direct capture is allowed
                 * - Push model is enabled (see NVFBC_CREATE_CAPTURE_SESSION_PARAMS::bPushModel)
                 * - The mouse cursor is not composited (see NVFBC_CREATE_CAPTURE_SESSION_PARAMS::bWithCursor)
                 * - No viewport transformation is required. This happens when the remote
                 *   desktop is e.g. rotated.

        */
        fprintf(stdout, "Setup: Creating NVFBC Capture Session...\n");
        NVFBC_CREATE_CAPTURE_SESSION_PARAMS cap_params;
        memset(&cap_params, 0, sizeof(cap_params));
        cap_params.dwVersion = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER;
        cap_params.eCaptureType = NVFBC_CAPTURE_TO_GL; // Specify capturing to GL
        //ConnectedPrimaryOutput,FirstConnectedOutput,EntireXScreen
        //NVFBC_TRACKING_DEFAULT,NVFBC_TRACKING_OUTPUT,NVFBC_TRACKING_SCREEN
        cap_params.eTrackingType = NVFBC_TRACKING_DEFAULT;//NVFBC_TRACKING_OUTPUT,NVFBC_TRACKING_DEFAULT; // Default tracking (screen changes)
        cap_params.bWithCursor = NVFBC_FALSE; // No cursor capture for minimal app
        cap_params.bDisableAutoModesetRecovery = NVFBC_TRUE; // Simpler, don't let NVFBC mess with modes
        cap_params.bRoundFrameSize = NVFBC_FALSE; // No rounding
        cap_params.dwSamplingRateMs = 33; // Aim for ~30 FPS
        cap_params.bPushModel = NVFBC_TRUE; // Non-blocking grabFrame
        cap_params.bAllowDirectCapture = NVFBC_TRUE; // Allow direct capture path
        cap_params.dwOutputId = 0x1bd;

        status = nvFBC_api_functions.nvFBCCreateCaptureSession(nvfbcApiHandle, &cap_params);

        if (status != NVFBC_SUCCESS) {
            fprintf(stderr, "Setup Error: nvFBCnvFBCCreateCaptureSession failed: %s\n",
                    nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
            break; // Exit setup block
        }
        nvfbc_capture_session_created = true;
        fprintf(stdout, "Setup: NVFBC Capture Session created (%lu).\n", nvfbcApiHandle);

        // --- NVFBC ToGL Setup ---
        // GLX context MUST be current for this call if bExternallyManagedContext is TRUE
        // It is already current.
        fprintf(stdout, "Setup: Setting up NVFBC ToGL capture...\n");
        memset(&togl_setup_params, 0, sizeof(togl_setup_params)); // Zero out the global setup params
        togl_setup_params.dwVersion = NVFBC_TOGL_SETUP_PARAMS_VER;
        togl_setup_params.eBufferFormat = NVFBC_BUFFER_FORMAT_BGRA; // Request texture ID output!
        togl_setup_params.bWithDiffMap = NVFBC_FALSE; // No diff map for minimal app

        status = nvFBC_api_functions.nvFBCToGLSetUp(nvfbcApiHandle, &togl_setup_params); // Use the API handle, not capture handle, for setup

        if (status != NVFBC_SUCCESS) {
            fprintf(stderr, "Setup Error: nvFBCToGLSetUp failed: %s\n",
                    nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
            break; // Exit setup block
        }
        fprintf(stdout, "Setup: NVFBC ToGL setup complete (GL_TEXTURE_ID).\n");

        // Unmake GLX context current after setup
        glXMakeCurrent(display, None, NULL);
        fprintf(stdout, "Setup: Unbound GLX context after ToGL setup.\n");

        // If we reach here, setup was successful
        fprintf(stdout, "Setup: All initialization steps completed successfully.\n");

    } while (false); // End of setup block


    // --- Main Loop ---
    // Only run the loop if setup was successful (check nvfbc_capture_session_created which is the last step)
    bool running = nvfbc_capture_session_created;
    unsigned int screen_texture_id = 0;
    int current_window_width = statusParams.screenSize.w > 0 ? statusParams.screenSize.w : 800;
    int current_window_height = statusParams.screenSize.h > 0 ? statusParams.screenSize.h : 600;


    if (running) {
         fprintf(stdout, "Entering main loop. Press any key or close window to exit.\n");
         // Map the window now that setup is complete
         XMapWindow(display, window);
         fprintf(stdout, "X11 window mapped.\n");

         // Initial OpenGL setup after window is mapped
         if (!glXMakeCurrent(display, window, glContext)) {
              fprintf(stderr, "Main Loop Error: Failed to make GLX context current for initial GL setup.\n");
              running = false; // Cannot render without context
         } else {
              // Basic OpenGL setup
              glEnable(GL_TEXTURE_2D);
              glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background

              // Initial viewport and projection
              glViewport(0, 0, current_window_width, current_window_height);
              glMatrixMode(GL_PROJECTION);
              glLoadIdentity();
              // Ortho projection mapping window coordinates (0,0)-(width,height)
              glOrtho(0.0, current_window_width, current_window_height, 0.0, -1.0, 1.0); // Y down
              glMatrixMode(GL_MODELVIEW);
              glLoadIdentity();

               // Unmake GLX context current before entering loop
             glXMakeCurrent(display, None, NULL);
         }

    } else {
        fprintf(stderr, "Main Loop: Setup failed, exiting.\n");
    }


    while (running) {
        // Handle X11 events
        while (XPending(display)) {
            XNextEvent(display, &event); // Using `event` defined globally
            switch (event.type) {
                case Expose:
                    // Window exposed, a redraw is needed. The main grab loop handles continuous redraws.
                    break;
                case ConfigureNotify:
                    // Window resized. Update GL viewport and projection based on *new* size.
                    current_window_width = event.xconfigure.width;
                    current_window_height = event.xconfigure.height;
                     if (glXMakeCurrent(display, window, glContext)) { // Make context current to update GL state
                         glViewport(0, 0, current_window_width, current_window_height);
                         glMatrixMode(GL_PROJECTION);
                         glLoadIdentity();
                         glOrtho(0.0, current_window_width, current_window_height, 0.0, -1.0, 1.0); // Y down
                         glMatrixMode(GL_MODELVIEW);
                         glLoadIdentity();
                          // Unmake GLX context current after updates
                         glXMakeCurrent(display, None, NULL);
                         fprintf(stdout, "Window resized to %dx%d\n", current_window_width, current_window_height);
                     } else {
                         fprintf(stderr, "Main Loop Error: Failed to make GLX context current for resize update.\n");
                         running = false; // Fatal? Or just skip redraw? Let's exit on fatal errors.
                     }

                    break;
                case KeyPress:
                    // running = false; // Exit on any key press
                    break;
                case ClientMessage: // Handle window close button
                    if ((Atom)event.xclient.data.l[0] == wmDeleteMessage) { // Use wmDeleteMessage defined earlier
                        running = false;
                    }
                    break;
            }
        }

        if (!running) break;

        // --- NVFBC Grab Frame (using pointer from struct) ---
        // Make GLX context current for NvFBC grab call (required with bExternallyManagedContext = TRUE)
        if (!glXMakeCurrent(display, window, glContext)) {
            fprintf(stderr, "Main Loop Error: Failed to make GLX context current for NVFBC grab! Fatal.\n");
            running = false;
            continue; // Exit loop via 'running' check
        }

        NVFBC_FRAME_GRAB_INFO info; // Frame info struct
        memset(&info, 0, sizeof(info));

        NVFBC_TOGL_GRAB_FRAME_PARAMS grab_params;
        memset(&grab_params, 0, sizeof(grab_params));
        grab_params.dwVersion = NVFBC_TOGL_GRAB_FRAME_PARAMS_VER;
        grab_params.dwFlags = NVFBC_TOGL_GRAB_FLAGS_NOWAIT | NVFBC_TOGL_GRAB_FLAGS_FORCE_REFRESH; // Non-blocking grab
        grab_params.pFrameGrabInfo = &info; // Pointer to receive frame info
        grab_params.dwTimeoutMs = 0; // Ignored with NOWAIT

        // Call the grab function using the API handle (from nvFBCCreateHandle)
        status = nvFBC_api_functions.nvFBCToGLGrabFrame(nvfbcApiHandle, &grab_params); // Use grabFrame from API list

        // Unmake GLX context current immediately after the NVFBC call
        glXMakeCurrent(display, None, NULL);


        bool frame_grabbed_this_loop = false;
        if (status == NVFBC_SUCCESS && info.bIsNewFrame == NVFBC_TRUE) {
            // Frame grabbed successfully and it's a new frame
            if (grab_params.dwTextureIndex < NVFBC_TOGL_TEXTURES_MAX) {
                 // Get the texture ID from the setup parameters using the index provided by grabParams
                 screen_texture_id = togl_setup_params.dwTextures[grab_params.dwTextureIndex];
                 frame_grabbed_this_loop = true;
                 // Frame dimensions are in info.dwWidth, info.dwHeight if needed
                 fprintf(stdout, "Grabbed new frame %ux%u, texture ID %u (index %u).\n",
                         info.dwWidth, info.dwHeight, screen_texture_id, grab_params.dwTextureIndex);
            } else {
                 fprintf(stderr, "Main Loop Warning: NvFBCToGLGrabFrame returned invalid texture index %u (max %u).\n",
                         grab_params.dwTextureIndex, NVFBC_TOGL_TEXTURES_MAX);
            }
        } else if (status != NVFBC_SUCCESS /*&& status != NVFBC_STATUS_NO_FRAME_AVAILABLE*/) {
             // Handle non-success, non-NO_FRAME_AVAILABLE errors
            fprintf(stderr, "Main Loop Error: nvFBCToGLGrabFrame failed: %s\n",
                    nvFBC_api_functions.nvFBCGetLastErrorStr ? nvFBC_api_functions.nvFBCGetLastErrorStr(nvfbcApiHandle) : "Unknown Error");
            running = false;
        }
        // NVFBC_STATUS_NO_FRAME_AVAILABLE or info.bIsNewFrame == NVFBC_FALSE is not an error, just no update.

        // --- OpenGL Rendering ---
        // Only render if we successfully grabbed a frame this loop AND got a valid texture ID
        if (frame_grabbed_this_loop && screen_texture_id != 0) {

            // Make GLX context current for drawing
            if (!glXMakeCurrent(display, window, glContext)) {
                fprintf(stderr, "Main Loop Error: Failed to make GLX context current for rendering! Fatal.\n");
                running = false;
                continue; // Exit loop via 'running' check
            }

            glClear(GL_COLOR_BUFFER_BIT);

            // Bind the texture provided by NVFBC
            glBindTexture(GL_TEXTURE_2D, screen_texture_id);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // or GL_NEAREST
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // or GL_NEAREST
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // or GL_REPEAT, GL_CLAMP_TO_BORDER etc.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // or GL_REPEAT etc.

            // Draw a quad covering the window, mapping the texture onto it
            glBegin(GL_QUADS);
                // Top-left
                glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
                // Top-right
                glTexCoord2f(1.0f, 0.0f); glVertex2f((float)current_window_width, 0.0f);
                // Bottom-right
                glTexCoord2f(1.0f, 1.0f); glVertex2f((float)current_window_width, (float)current_window_height);
                // Bottom-left
                glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (float)current_window_height);
            glEnd();

            // Swap buffers to show the rendered frame
            glXSwapBuffers(display, window);

            // Unbind texture
            glBindTexture(GL_TEXTURE_2D, 0);

            // Unmake GLX context current after drawing
            glXMakeCurrent(display, None, NULL);

        } else {
             // If no new frame was grabbed, add a small sleep to avoid busy-waiting
             // In a push model with NOWAIT, this is important to not spin constantly.
             usleep(1000); // Sleep 1ms
        }
    } // End of main loop

    fprintf(stdout, "Main Loop exited.\n");

    // atexit(cleanup_resources) will be called automatically when main returns

    return 0;
}