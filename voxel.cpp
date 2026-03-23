#pragma once
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

constexpr int MAP_SIZE = 1024; // must stay power-of-two for fast wrapping
constexpr float PI = 3.1415926535f;

struct AppState {
    int width = 1280;
    int height = 720;

    std::vector<uint32_t> pixels;
    std::vector<int> columnTop;

    std::vector<uint8_t> heightMap;
    std::vector<uint32_t> colorMap;

    BITMAPINFO bmi{};

    float camX = 512.0f;
    float camY = 512.0f;
    float camZ = 90.0f;
    float angle = 0.0f;

    bool running = true;
    bool flyMode = false;

    float fpsTimer = 0.0f;
    int fpsFrames = 0;
} gApp;

static inline float Lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float Smooth(float t) {
    return t * t * (3.0f - 2.0f * t);
}

static inline uint32_t MakeColor(uint8_t r, uint8_t g, uint8_t b) {
    return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
}

static inline uint8_t ClampByte(int v) {
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

static inline uint32_t Hash2D(int x, int y) {
    uint32_t h = 2166136261u;
    h = (h ^ uint32_t(x)) * 16777619u;
    h = (h ^ uint32_t(y)) * 16777619u;
    h ^= (h >> 13);
    h *= 1274126177u;
    h ^= (h >> 16);
    return h;
}

static inline float Random01(int x, int y) {
    return float(Hash2D(x, y) & 0x00FFFFFF) / float(0x00FFFFFF);
}

float ValueNoise(float x, float y) {
    int x0 = int(std::floor(x));
    int y0 = int(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float tx = x - float(x0);
    float ty = y - float(y0);

    float sx = Smooth(tx);
    float sy = Smooth(ty);

    float v00 = Random01(x0, y0);
    float v10 = Random01(x1, y0);
    float v01 = Random01(x0, y1);
    float v11 = Random01(x1, y1);

    float a = Lerp(v00, v10, sx);
    float b = Lerp(v01, v11, sx);
    return Lerp(a, b, sy);
}

float FBM(float x, float y, int octaves = 6) {
    float sum = 0.0f;
    float amp = 0.5f;
    float freq = 1.0f;
    float norm = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        sum += ValueNoise(x * freq, y * freq) * amp;
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }

    return sum / norm;
}

static inline int WrapCoord(int v) {
    return v & (MAP_SIZE - 1);
}

static inline int MapIndex(int x, int y) {
    return WrapCoord(y) * MAP_SIZE + WrapCoord(x);
}

static inline uint8_t SampleHeightNearest(float x, float y) {
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    return gApp.heightMap[MapIndex(ix, iy)];
}

static inline uint32_t SampleColorNearest(float x, float y) {
    int ix = int(std::floor(x));
    int iy = int(std::floor(y));
    return gApp.colorMap[MapIndex(ix, iy)];
}

uint32_t ApplyFog(uint32_t color, float t) {
    t = std::clamp(t, 0.0f, 1.0f);

    const uint8_t fogR = 170;
    const uint8_t fogG = 205;
    const uint8_t fogB = 235;

    int r = (color >> 16) & 255;
    int g = (color >> 8) & 255;
    int b = color & 255;

    r = int(Lerp(float(r), float(fogR), t));
    g = int(Lerp(float(g), float(fogG), t));
    b = int(Lerp(float(b), float(fogB), t));

    return MakeColor(ClampByte(r), ClampByte(g), ClampByte(b));
}

uint32_t Darken(uint32_t color, float factor) {
    factor = std::clamp(factor, 0.0f, 1.0f);

    int r = int(((color >> 16) & 255) * factor);
    int g = int(((color >> 8) & 255) * factor);
    int b = int((color & 255) * factor);

    return MakeColor(ClampByte(r), ClampByte(g), ClampByte(b));
}

void ResizeBackbuffer(int width, int height) {
    gApp.width = std::max(1, width);
    gApp.height = std::max(1, height);

    gApp.pixels.assign(gApp.width * gApp.height, 0);
    gApp.columnTop.assign(gApp.width, gApp.height - 1);

    ZeroMemory(&gApp.bmi, sizeof(gApp.bmi));
    gApp.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    gApp.bmi.bmiHeader.biWidth = gApp.width;
    gApp.bmi.bmiHeader.biHeight = -gApp.height; // top-down
    gApp.bmi.bmiHeader.biPlanes = 1;
    gApp.bmi.bmiHeader.biBitCount = 32;
    gApp.bmi.bmiHeader.biCompression = BI_RGB;
}

void GenerateTerrain() {
    gApp.heightMap.resize(MAP_SIZE * MAP_SIZE);
    gApp.colorMap.resize(MAP_SIZE * MAP_SIZE);

    for (int y = 0; y < MAP_SIZE; ++y) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            float nx = x * 0.0035f;
            float ny = y * 0.0035f;

            float large = FBM(x * 0.0012f, y * 0.0012f, 5);
            float medium = FBM(nx, ny, 6);
            float detail = FBM(x * 0.012f, y * 0.012f, 4);

            float e = 0.55f * medium + 0.30f * large + 0.15f * detail;
            e = std::pow(e, 1.35f);

            float ridge = std::fabs(FBM(x * 0.006f, y * 0.006f, 5) - 0.5f) * 2.0f;
            float heightF = 20.0f + e * 170.0f + ridge * 18.0f;

            int h = std::clamp(int(heightF), 0, 255);
            gApp.heightMap[y * MAP_SIZE + x] = static_cast<uint8_t>(h);

            uint32_t c = 0;

            if (h < 42) {
                c = MakeColor(10, 30, 90); // deep water
            }
            else if (h < 52) {
                c = MakeColor(20, 70, 140); // shallow water
            }
            else if (h < 58) {
                c = MakeColor(194, 178, 128); // beach
            }
            else if (h < 95) {
                int green = 110 + int(detail * 40.0f);
                c = MakeColor(35, ClampByte(green), 45); // low grass
            }
            else if (h < 140) {
                int green = 95 + int(detail * 30.0f);
                c = MakeColor(50, ClampByte(green), 40); // upland grass
            }
            else if (h < 180) {
                int rock = 95 + int(detail * 35.0f);
                c = MakeColor(ClampByte(rock), ClampByte(rock), ClampByte(rock - 8)); // rock
            }
            else {
                int snow = 220 + int(detail * 25.0f);
                c = MakeColor(ClampByte(snow), ClampByte(snow), ClampByte(snow)); // snow
            }

            gApp.colorMap[y * MAP_SIZE + x] = c;
        }
    }
}

void ClearSky() {
    for (int y = 0; y < gApp.height; ++y) {
        float t = float(y) / float(std::max(1, gApp.height - 1));

        uint8_t r = ClampByte(int(Lerp(75.0f, 185.0f, t)));
        uint8_t g = ClampByte(int(Lerp(120.0f, 220.0f, t)));
        uint8_t b = ClampByte(int(Lerp(185.0f, 250.0f, t)));

        uint32_t color = MakeColor(r, g, b);
        uint32_t* row = &gApp.pixels[y * gApp.width];
        for (int x = 0; x < gApp.width; ++x) {
            row[x] = color;
        }
    }

    std::fill(gApp.columnTop.begin(), gApp.columnTop.end(), gApp.height - 1);
}

void DrawVerticalLine(int x, int yTop, int yBottom, uint32_t color) {
    if (x < 0 || x >= gApp.width) return;

    yTop = std::max(yTop, 0);
    yBottom = std::min(yBottom, gApp.height - 1);
    if (yTop > yBottom) return;

    uint32_t* dst = &gApp.pixels[yTop * gApp.width + x];
    for (int y = yTop; y <= yBottom; ++y) {
        *dst = color;
        dst += gApp.width;
    }
}

void RenderVoxelTerrain() {
    ClearSky();

    const float horizon = gApp.height * 0.55f;
    const float tanHalfFov = 0.95f;
    const float maxDistance = 800.0f;
    const float projectionScale = 260.0f;

    float sinA = std::sin(gApp.angle);
    float cosA = std::cos(gApp.angle);

    float zStep = 1.0f;

    for (float z = 1.0f; z < maxDistance; z += zStep) {
        float centerX = gApp.camX + cosA * z;
        float centerY = gApp.camY + sinA * z;

        float span = z * tanHalfFov;

        float leftX = centerX - sinA * span;
        float leftY = centerY + cosA * span;

        float rightX = centerX + sinA * span;
        float rightY = centerY - cosA * span;

        float dx = (rightX - leftX) / float(gApp.width);
        float dy = (rightY - leftY) / float(gApp.width);

        float wx = leftX;
        float wy = leftY;

        float invZ = 1.0f / z;
        float fog = (z / maxDistance);
        fog *= fog;

        for (int x = 0; x < gApp.width; ++x) {
            if (gApp.columnTop[x] < 0) {
                wx += dx;
                wy += dy;
                continue;
            }

            uint8_t terrainHeight = SampleHeightNearest(wx, wy);
            uint32_t terrainColor = SampleColorNearest(wx, wy);

            float projectedY = horizon - ((float(terrainHeight) - gApp.camZ) * projectionScale * invZ);
            int screenY = int(projectedY);

            if (screenY < gApp.columnTop[x]) {
                uint8_t nextH = SampleHeightNearest(wx + dx, wy + dy);
                float slopeShade = 0.85f + (float(nextH) - float(terrainHeight)) * 0.01f;
                slopeShade = std::clamp(slopeShade, 0.55f, 1.10f);

                terrainColor = Darken(terrainColor, slopeShade);
                terrainColor = ApplyFog(terrainColor, fog);

                DrawVerticalLine(x, screenY, gApp.columnTop[x], terrainColor);
                gApp.columnTop[x] = screenY - 1;
            }

            wx += dx;
            wy += dy;
        }

        zStep += 0.0065f; // progressive stepping for speed
    }
}

void Present(HWND hwnd) {
    HDC dc = GetDC(hwnd);

    StretchDIBits(
        dc,
        0, 0, gApp.width, gApp.height,
        0, 0, gApp.width, gApp.height,
        gApp.pixels.data(),
        &gApp.bmi,
        DIB_RGB_COLORS,
        SRCCOPY
    );

    ReleaseDC(hwnd, dc);
}

bool KeyDown(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

void WrapCamera() {
    while (gApp.camX < 0.0f) gApp.camX += float(MAP_SIZE);
    while (gApp.camY < 0.0f) gApp.camY += float(MAP_SIZE);
    while (gApp.camX >= float(MAP_SIZE)) gApp.camX -= float(MAP_SIZE);
    while (gApp.camY >= float(MAP_SIZE)) gApp.camY -= float(MAP_SIZE);
}

void UpdateCamera(float dt) {
    float moveSpeed = 90.0f;
    float strafeSpeed = 75.0f;
    float turnSpeed = 1.7f;
    float liftSpeed = 65.0f;

    if (KeyDown(VK_SHIFT)) {
        moveSpeed *= 2.0f;
        strafeSpeed *= 2.0f;
        liftSpeed *= 2.0f;
    }

    if (KeyDown(VK_LEFT) || KeyDown('A')) gApp.angle -= turnSpeed * dt;
    if (KeyDown(VK_RIGHT) || KeyDown('D')) gApp.angle += turnSpeed * dt;

    float forward = 0.0f;
    float strafe = 0.0f;

    if (KeyDown('W') || KeyDown(VK_UP)) forward += 1.0f;
    if (KeyDown('S') || KeyDown(VK_DOWN)) forward -= 1.0f;
    if (KeyDown('Q')) strafe -= 1.0f;
    if (KeyDown('E')) strafe += 1.0f;

    float sinA = std::sin(gApp.angle);
    float cosA = std::cos(gApp.angle);

    gApp.camX += cosA * forward * moveSpeed * dt;
    gApp.camY += sinA * forward * moveSpeed * dt;

    gApp.camX += -sinA * strafe * strafeSpeed * dt;
    gApp.camY += cosA * strafe * strafeSpeed * dt;

    if (KeyDown('T')) gApp.flyMode = true;
    if (KeyDown('G')) gApp.flyMode = false;

    if (gApp.flyMode) {
        if (KeyDown('R')) gApp.camZ += liftSpeed * dt;
        if (KeyDown('F')) gApp.camZ -= liftSpeed * dt;
    }
    else {
        float ground = float(SampleHeightNearest(gApp.camX, gApp.camY)) + 18.0f;
        float follow = std::clamp(dt * 8.0f, 0.0f, 1.0f);
        gApp.camZ = Lerp(gApp.camZ, ground, follow);
    }

    WrapCamera();
}

void UpdateWindowTitle(HWND hwnd, float dt) {
    gApp.fpsTimer += dt;
    gApp.fpsFrames++;

    if (gApp.fpsTimer >= 0.5f) {
        float fps = float(gApp.fpsFrames) / gApp.fpsTimer;
        std::wstring title =
            L"Scratch Voxel Terrain Renderer | "
            L"W/S move, A/D turn, Q/E strafe, T fly on, G fly off, R/F vertical | FPS: " +
            std::to_wstring(int(fps));

        SetWindowTextW(hwnd, title.c_str());
        gApp.fpsTimer = 0.0f;
        gApp.fpsFrames = 0;
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_SIZE: {
        int w = LOWORD(lParam);
        int h = HIWORD(lParam);
        if (w > 0 && h > 0) {
            ResizeBackbuffer(w, h);
        }
        return 0;
    }
    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        gApp.running = false;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}