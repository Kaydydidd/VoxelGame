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

// ---------------------------------------------------------------------------
//  World constants
// ---------------------------------------------------------------------------
inline constexpr int   MAP_SIZE = 1024;   // horizontal extent (power-of-two)
inline constexpr int   MAP_HEIGHT = 256;    // vertical extent
inline constexpr float PI = 3.1415926535f;

// ---------------------------------------------------------------------------
//  Block types – each voxel stores one of these
// ---------------------------------------------------------------------------
enum BlockType : uint8_t {
    BLOCK_AIR = 0,
    BLOCK_STONE = 1,
    BLOCK_DIRT = 2,
    BLOCK_GRASS = 3,
    BLOCK_SAND = 4,
    BLOCK_WATER = 5,
    BLOCK_SNOW = 6,
    BLOCK_DEEP_WATER = 7,
    BLOCK_ROCK = 8,
    BLOCK_COUNT
};

// ---------------------------------------------------------------------------
//  3-D block index   (X fastest, then Y, then Z)
//  Block at world integer coordinate (x, y, z) occupies unit cube [x,x+1)…
// ---------------------------------------------------------------------------
inline int BlockIndex(int x, int y, int z) {
    return x + y * MAP_SIZE + z * MAP_SIZE * MAP_HEIGHT;
}

// ---------------------------------------------------------------------------
//  Application state  – shared between Demo.cpp and voxel.cpp
//  D3D12 resources live in file-static storage inside voxel.cpp.
// ---------------------------------------------------------------------------
struct AppState {
    int  width = 1280;
    int  height = 720;
    HWND hwnd = nullptr;

    // 3-D block grid  (MAP_SIZE × MAP_HEIGHT × MAP_SIZE)
    std::vector<uint8_t> blockMap;

    // 2-D surface-height cache for CPU camera follow (MAP_SIZE × MAP_SIZE)
    std::vector<uint8_t> heightMap;

    // Camera
    float camX = 512.0f;          // world X
    float camY = 512.0f;          // world Z  (second horizontal axis)
    float camZ = 90.0f;           // world Y  (altitude)
    float angle = 0.0f;            // yaw   (radians)
    float pitch = 0.0f;            // pitch (radians)

    bool running = true;
    bool flyMode = false;
    bool mouseCaptured = false;

    float fpsTimer = 0.0f;
    int   fpsFrames = 0;
};

extern AppState gApp;

// ---------------------------------------------------------------------------
//  Public interface consumed by Demo.cpp
// ---------------------------------------------------------------------------
void GenerateTerrain();
void InitD3D12(HWND hwnd);
void ShutdownD3D12();
void Render();
void ResizeBackbuffer(int width, int height);
void UpdateCamera(float dt);
void UpdateWindowTitle(HWND hwnd, float dt);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);