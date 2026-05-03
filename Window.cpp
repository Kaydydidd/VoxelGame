#include "voxel.h"
#include "ChunkStreaming.h"

#include <cstdio>

void UpdateWindowTitle(HWND hwnd, float dt) {
    gApp.fpsTimer += dt; gApp.fpsFrames++;
    if (gApp.fpsTimer >= 0.5f) {
        float fps = float(gApp.fpsFrames) / gApp.fpsTimer;
        int loaded = int(LoadedChunkCount());
        int queued = int(UploadQueueCount());
        wchar_t buf[256];
        swprintf_s(buf, L"Voxel Terrain | chunks:%d queue:%d | FPS:%d", loaded, queued, int(fps));
        SetWindowTextW(hwnd, buf);
        gApp.fpsTimer = 0; gApp.fpsFrames = 0;
    }
}

static void CenterCursor() {
    RECT r; GetClientRect(gApp.hwnd, &r);
    POINT c = { (r.right - r.left) / 2, (r.bottom - r.top) / 2 };
    ClientToScreen(gApp.hwnd, &c);
    SetCursorPos(c.x, c.y);
}

// ===================================================================
//  Window procedure
// ===================================================================
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_SIZE: {
        int w = LOWORD(lParam), h = HIWORD(lParam);
        if (w > 0 && h > 0) ResizeBackbuffer(w, h);
        return 0;
    }
    case WM_LBUTTONDOWN:
        if (!gApp.mouseCaptured) {
            gApp.mouseCaptured = true;
            ShowCursor(FALSE);
            CenterCursor();
        }
        return 0;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE && gApp.mouseCaptured) {
            gApp.mouseCaptured = false;
            ShowCursor(TRUE);
            return 0;
        }
        break;
    case WM_KILLFOCUS:
        if (gApp.mouseCaptured) {
            gApp.mouseCaptured = false;
            ShowCursor(TRUE);
        }
        return 0;
    case WM_CLOSE:   DestroyWindow(hwnd); return 0;
    case WM_DESTROY: gApp.running = false; PostQuitMessage(0); return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}