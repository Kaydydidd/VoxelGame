#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

inline constexpr int MAP_SIZE = 1024;
inline constexpr float PI = 3.1415926535f;

struct AppState {
    int width = 2560;
    int height = 1440;

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
};

extern AppState gApp;

float ValueNoise(float x, float y);
float FBM(float x, float y, int octaves = 6);
uint32_t ApplyFog(uint32_t color, float t);
uint32_t Darken(uint32_t color, float factor);

void ResizeBackbuffer(int width, int height);
void GenerateTerrain();
void ClearSky();
void DrawVerticalLine(int x, int yTop, int yBottom, uint32_t color);
void RenderVoxelTerrain();
void Present(HWND hwnd);
bool KeyDown(int vk);
void WrapCamera();
void UpdateCamera(float dt);
void UpdateWindowTitle(HWND hwnd, float dt);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
int RunVoxelTerrainDemo(HINSTANCE hInstance, int nCmdShow);