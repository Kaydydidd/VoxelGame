#include "voxel.h"
#include "ChunkStreaming.h"

#include <algorithm>
#include <cmath>



static float Lerp(float a, float b, float t) { return a + (b - a) * t; }

// ===================================================================
//  Camera / input
// ===================================================================
static bool KeyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

static void CenterCursor() {
    RECT r; GetClientRect(gApp.hwnd, &r);
    POINT c = { (r.right - r.left) / 2, (r.bottom - r.top) / 2 };
    ClientToScreen(gApp.hwnd, &c);
    SetCursorPos(c.x, c.y);
}

void UpdateCamera(float dt) {
    float moveSpeed = 8.0f, strafeSpeed = 7.0f, liftSpeed = 8.0f;
    if (KeyDown(VK_SHIFT)) { moveSpeed *= 3; strafeSpeed *= 3; liftSpeed *= 3; }

    if (gApp.mouseCaptured) {
        RECT cr; GetClientRect(gApp.hwnd, &cr);
        POINT sc = { (cr.right - cr.left) / 2, (cr.bottom - cr.top) / 2 };
        ClientToScreen(gApp.hwnd, &sc);
        POINT cur; GetCursorPos(&cur);
        float dx = float(cur.x - sc.x), dy = float(cur.y - sc.y);
        gApp.angle += dx * 0.002f;
        gApp.pitch -= dy * 0.002f;
        gApp.pitch = std::clamp(gApp.pitch, -1.48f, 1.48f);
        SetCursorPos(sc.x, sc.y);
    }
    if (KeyDown(VK_LEFT))  gApp.angle -= 1.7f * dt;
    if (KeyDown(VK_RIGHT)) gApp.angle += 1.7f * dt;

    float fwd = 0, str = 0;
    if (KeyDown('W') || KeyDown(VK_UP))   fwd += 1;
    if (KeyDown('S') || KeyDown(VK_DOWN)) fwd -= 1;
    if (KeyDown('A') || KeyDown('Q'))     str -= 1;
    if (KeyDown('D') || KeyDown('E'))     str += 1;

    float cosA = std::cos(gApp.angle), sinA = std::sin(gApp.angle);
    gApp.camX += cosA * fwd * moveSpeed * dt;
    gApp.camY += sinA * fwd * moveSpeed * dt;
    gApp.camX += -sinA * str * strafeSpeed * dt;
    gApp.camY += cosA * str * strafeSpeed * dt;

    if (KeyDown('T')) gApp.flyMode = true;
    if (KeyDown('G')) gApp.flyMode = false;

    if (gApp.flyMode) {
        if (KeyDown(VK_SPACE) || KeyDown('R')) gApp.camZ += liftSpeed * dt;
        if (KeyDown(VK_CONTROL) || KeyDown('F')) gApp.camZ -= liftSpeed * dt;
    }
    else {
        float sh = float(SurfaceHeightAt(gApp.camX, gApp.camY));
        float target = sh + 1.0f + 1.5f;
        float follow = std::clamp(dt * 8.0f, 0.0f, 1.0f);
        gApp.camZ = Lerp(gApp.camZ, target, follow);
    }
}