#define NOMINMAX
#include <windows.h>
#include <chrono>

#include "voxel.h"

int RunVoxelTerrainDemo(HINSTANCE hInstance, int nCmdShow);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    return RunVoxelTerrainDemo(hInstance, nCmdShow);
}

int RunVoxelTerrainDemo(HINSTANCE hInstance, int nCmdShow) {
    const wchar_t CLASS_NAME[] = L"ScratchVoxelRendererWindowClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.style = CS_HREDRAW | CS_VREDRAW;

    RegisterClassW(&wc);

    RECT rect{ 0, 0, gApp.width, gApp.height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowExW(
        0,
        CLASS_NAME,
        L"Scratch Voxel Terrain Renderer",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hwnd) return 0;

    // WM_SIZE has already fired during CreateWindowExW and stashed the
    // client dimensions via ResizeBackbuffer's no-device early-out.

    // Order matters: terrain must exist before InitD3D12 uploads it.
    GenerateTerrain();
    InitD3D12(hwnd);          // device, queue, swap chain, PSO, terrain upload

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    auto lastTime = std::chrono::high_resolution_clock::now();

    MSG msg{};
    gApp.running = true;
    while (gApp.running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) gApp.running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!gApp.running) break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        dt = std::min(dt, 0.033f);

        UpdateCamera(dt);          // CPU simulation
        Render();                  // record + submit + present (GPU does all shading)
        UpdateWindowTitle(hwnd, dt);
    }

    ShutdownD3D12();               // drain queue, release handles
    return 0;
}