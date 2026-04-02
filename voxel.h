#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <string>


// ---------------------------------------------------------------------------
//  World / chunk constants
// ---------------------------------------------------------------------------
inline constexpr int   CHUNK_X = 32;
inline constexpr int   CHUNK_Z = 32;
inline constexpr int   CHUNK_Y = 128;
inline constexpr int   SECTION_Y = 16;
inline constexpr int   SECTIONS_PER_CHUNK = CHUNK_Y / SECTION_Y;   // 8
inline constexpr int   REGION_CHUNKS = 8;                       // chunks per region side
inline constexpr int   LOAD_REGIONS = 3;                       // 3×3 regions loaded
inline constexpr int   LOAD_CHUNKS = LOAD_REGIONS * REGION_CHUNKS; // 24 per axis
inline constexpr int   ATLAS_XZ = LOAD_CHUNKS * CHUNK_X;   // 768
inline constexpr float PI = 3.1415926535f;

// ---------------------------------------------------------------------------
//  Block types
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
//  Application state – only fields Demo.cpp touches.
//  Chunk data and GPU resources live in voxel.cpp file scope.
// ---------------------------------------------------------------------------
struct AppState {
    int  width = 1280;
    int  height = 720;
    HWND hwnd = nullptr;

    float camX = 512.0f;
    float camY = 512.0f;
    float camZ = 90.0f;
    float angle = 0.0f;
    float pitch = 0.0f;

    bool running = true;
    bool flyMode = false;
    bool mouseCaptured = false;

    float fpsTimer = 0.0f;
    int   fpsFrames = 0;
};

extern AppState gApp;

// ---------------------------------------------------------------------------
//  Public interface (signatures match Demo.cpp call sites)
// ---------------------------------------------------------------------------
void GenerateTerrain();
void InitD3D12(HWND hwnd);
void ShutdownD3D12();
void Render();
void ResizeBackbuffer(int width, int height);
void UpdateCamera(float dt);
void UpdateWindowTitle(HWND hwnd, float dt);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);