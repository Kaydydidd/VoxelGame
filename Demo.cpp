#define NOMINMAX
#include <windows.h>
#include "voxel.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    return RunVoxelTerrainDemo(hInstance, nCmdShow);
}

// Implemented in VoxelTerrainRenderer.cpp
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
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (!hwnd) return 0;

    ResizeBackbuffer(gApp.width, gApp.height);
    GenerateTerrain();

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    auto lastTime = std::chrono::high_resolution_clock::now();

    MSG msg{};
    while (gApp.running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                gApp.running = false;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        dt = std::min(dt, 0.033f); // avoid huge jumps

        UpdateCamera(dt);
        RenderVoxelTerrain();
        Present(hwnd);
        UpdateWindowTitle(hwnd, dt);
    }

    return 0;
}