#pragma once
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

constexpr int   MAP_SIZE = 1024; // must stay power-of-two for fast wrapping
constexpr float PI = 3.1415926535f;
constexpr UINT  FRAME_COUNT = 2;
constexpr UINT  CB_ALIGN = 256;

// ---------------------------------------------------------------------------
//  Frame constants mirrored in HLSL
// ---------------------------------------------------------------------------
struct FrameConstants {
    float    camX, camY, camZ, angle;
    uint32_t width, height, mapSize, _pad;
};

// ---------------------------------------------------------------------------
//  Application state
// ---------------------------------------------------------------------------
struct AppState {
    int  width = 1280;
    int  height = 720;
    HWND hwnd = nullptr;

    // CPU-side terrain. heightMap kept for camera ground-follow (simulation).
    // colorMap freed after GPU upload.
    std::vector<uint8_t>  heightMap;
    std::vector<uint32_t> colorMap;

    float camX = 512.0f;
    float camY = 512.0f;
    float camZ = 90.0f;
    float angle = 0.0f;

    bool running = true;
    bool flyMode = false;

    float fpsTimer = 0.0f;
    int   fpsFrames = 0;

    // --- D3D12 core ---
    ComPtr<ID3D12Device>              device;
    ComPtr<ID3D12CommandQueue>        queue;
    ComPtr<IDXGISwapChain3>           swapChain;
    ComPtr<ID3D12CommandAllocator>    cmdAlloc[FRAME_COUNT];
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence>               fence;
    HANDLE                            fenceEvent = nullptr;
    UINT64                            fenceValues[FRAME_COUNT]{};
    UINT                              frameIndex = 0;

    // --- descriptors ---
    ComPtr<ID3D12DescriptorHeap> srvHeap;   // t0,t1,u0
    UINT                         srvDescSize = 0;

    // --- pipeline ---
    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> pso;

    // --- resources ---
    ComPtr<ID3D12Resource> heightTex;   // DEFAULT, immutable
    ComPtr<ID3D12Resource> colorTex;    // DEFAULT, immutable
    ComPtr<ID3D12Resource> outputTex;   // DEFAULT, UAV, resize-bound
    ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT];
    ComPtr<ID3D12Resource> cbUpload;    // UPLOAD, persistently mapped ring
    uint8_t* cbMapped = nullptr;
} gApp;

// ---------------------------------------------------------------------------
//  Error handling
// ---------------------------------------------------------------------------
static inline void Check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        OutputDebugStringA(what);
        OutputDebugStringA("\n");
        throw std::runtime_error(what);
    }
}

// ---------------------------------------------------------------------------
//  Math / noise  (unchanged – CPU-side terrain authoring only)
// ---------------------------------------------------------------------------
static inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }
static inline float Smooth(float t) { return t * t * (3.0f - 2.0f * t); }

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
    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    int x1 = x0 + 1, y1 = y0 + 1;
    float sx = Smooth(x - float(x0));
    float sy = Smooth(y - float(y0));
    float a = Lerp(Random01(x0, y0), Random01(x1, y0), sx);
    float b = Lerp(Random01(x0, y1), Random01(x1, y1), sx);
    return Lerp(a, b, sy);
}

float FBM(float x, float y, int octaves = 6) {
    float sum = 0, amp = 0.5f, freq = 1, norm = 0;
    for (int i = 0; i < octaves; ++i) {
        sum += ValueNoise(x * freq, y * freq) * amp;
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return sum / norm;
}

static inline int WrapCoord(int v) { return v & (MAP_SIZE - 1); }
static inline int MapIndex(int x, int y) {
    return WrapCoord(y) * MAP_SIZE + WrapCoord(x);
}

// Kept for camera ground-follow (simulation, not rendering).
static inline uint8_t SampleHeightNearest(float x, float y) {
    return gApp.heightMap[MapIndex(int(std::floor(x)), int(std::floor(y)))];
}

// ---------------------------------------------------------------------------
//  Terrain generation  (unchanged – runs once, CPU authoring)
// ---------------------------------------------------------------------------
void GenerateTerrain() {
    gApp.heightMap.resize(MAP_SIZE * MAP_SIZE);
    gApp.colorMap.resize(MAP_SIZE * MAP_SIZE);

    for (int y = 0; y < MAP_SIZE; ++y) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            float nx = x * 0.0035f, ny = y * 0.0035f;
            float large = FBM(x * 0.0012f, y * 0.0012f, 5);
            float medium = FBM(nx, ny, 6);
            float detail = FBM(x * 0.012f, y * 0.012f, 4);

            float e = 0.55f * medium + 0.30f * large + 0.15f * detail;
            e = std::pow(e, 1.35f);

            float ridge = std::fabs(FBM(x * 0.006f, y * 0.006f, 5) - 0.5f) * 2.0f;
            float heightF = 20.0f + e * 170.0f + ridge * 18.0f;

            int h = std::clamp(int(heightF), 0, 255);
            gApp.heightMap[y * MAP_SIZE + x] = static_cast<uint8_t>(h);

            uint32_t c;
            if (h < 42)  c = MakeColor(10, 30, 90);
            else if (h < 52)  c = MakeColor(20, 70, 140);
            else if (h < 58)  c = MakeColor(194, 178, 128);
            else if (h < 95) { int g = 110 + int(detail * 40); c = MakeColor(35, ClampByte(g), 45); }
            else if (h < 140) { int g = 95 + int(detail * 30); c = MakeColor(50, ClampByte(g), 40); }
            else if (h < 180) { int r = 95 + int(detail * 35); c = MakeColor(ClampByte(r), ClampByte(r), ClampByte(r - 8)); }
            else { int s = 220 + int(detail * 25); c = MakeColor(ClampByte(s), ClampByte(s), ClampByte(s)); }

            gApp.colorMap[y * MAP_SIZE + x] = c;
        }
    }
}

// ---------------------------------------------------------------------------
//  HLSL compute shader  – direct transliteration of RenderVoxelTerrain
//  Integer colour math preserved to match CPU truncation exactly.
// ---------------------------------------------------------------------------
static const char* g_shaderSrc = R"(
cbuffer FrameConstants : register(b0)
{
    float  camX, camY, camZ, angle;
    uint   screenWidth, screenHeight, mapSize, _pad;
};

Texture2D<uint>   HeightMap : register(t0); // R8_UINT  0..255
Texture2D<uint>   ColorMap  : register(t1); // R32_UINT 0x00RRGGBB
RWTexture2D<float4> Output  : register(u0);

uint  Wrap(int v)            { return uint(v) & (mapSize - 1u); }
uint  SampleHeight(float x, float y)
{
    int ix = int(floor(x)), iy = int(floor(y));
    return HeightMap.Load(int3(Wrap(ix), Wrap(iy), 0));
}
uint  SampleColor(float x, float y)
{
    int ix = int(floor(x)), iy = int(floor(y));
    return ColorMap.Load(int3(Wrap(ix), Wrap(iy), 0));
}
uint3 Unpack(uint c) { return uint3((c>>16)&255u, (c>>8)&255u, c&255u); }

uint3 Darken(uint3 c, float f)
{
    f = saturate(f);
    return uint3(uint(float(c.r)*f), uint(float(c.g)*f), uint(float(c.b)*f));
}
uint3 ApplyFog(uint3 c, float t)
{
    t = saturate(t);
    const float3 fog = float3(170,205,235);
    float3 fc = float3(c);
    return uint3(uint(lerp(fc.r,fog.r,t)),
                 uint(lerp(fc.g,fog.g,t)),
                 uint(lerp(fc.b,fog.b,t)));
}
float4 ToOut(uint3 c) { return float4(float3(c)/255.0, 1.0); }

[numthreads(64,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x;
    if (x >= screenWidth) return;

    // ---- ClearSky (per-column) ----
    for (uint sy = 0; sy < screenHeight; ++sy)
    {
        float t = float(sy) / float(max(1u, screenHeight - 1u));
        uint r = uint(lerp(75.0,  185.0, t));
        uint g = uint(lerp(120.0, 220.0, t));
        uint b = uint(lerp(185.0, 250.0, t));
        Output[uint2(x, sy)] = ToOut(uint3(r,g,b));
    }

    int columnTop = int(screenHeight) - 1;

    const float horizon         = float(screenHeight) * 0.55f;
    const float tanHalfFov      = 0.95f;
    const float maxDistance     = 800.0f;
    const float projectionScale = 260.0f;

    float sinA = sin(angle);
    float cosA = cos(angle);

    float zStep = 1.0f;

    [loop]
    for (float z = 1.0f; z < maxDistance; z += zStep)
    {
        float centerX = camX + cosA * z;
        float centerY = camY + sinA * z;
        float span    = z * tanHalfFov;

        float leftX  = centerX - sinA * span;
        float leftY  = centerY + cosA * span;
        float rightX = centerX + sinA * span;
        float rightY = centerY - cosA * span;

        float dx = (rightX - leftX) / float(screenWidth);
        float dy = (rightY - leftY) / float(screenWidth);

        float wx = leftX + dx * float(x);
        float wy = leftY + dy * float(x);

        float invZ = 1.0f / z;
        float fog  = z / maxDistance; fog *= fog;

        if (columnTop >= 0)
        {
            uint  th = SampleHeight(wx, wy);
            uint  tc = SampleColor (wx, wy);

            float projY = horizon - ((float(th) - camZ) * projectionScale * invZ);
            int   scrY  = int(projY);

            if (scrY < columnTop)
            {
                uint  nh = SampleHeight(wx + dx, wy + dy);
                float slope = 0.85f + (float(nh) - float(th)) * 0.01f;
                slope = clamp(slope, 0.55f, 1.10f);

                uint3 col = Unpack(tc);
                col = Darken(col, slope);
                col = ApplyFog(col, fog);

                int yTop = max(scrY, 0);
                int yBot = min(columnTop, int(screenHeight) - 1);
                [loop]
                for (int py = yTop; py <= yBot; ++py)
                    Output[uint2(x, uint(py))] = ToOut(col);

                columnTop = scrY - 1;
            }
        }

        zStep += 0.0065f;
    }
}
)";

// ---------------------------------------------------------------------------
//  D3D12 helpers
// ---------------------------------------------------------------------------
static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

// ---------------------------------------------------------------------------
//  GPU synchronisation
//  Invariant: fenceValues[i] is always the NEXT value to be signalled for
//  slot i. It is strictly greater than any value already on the fence.
// ---------------------------------------------------------------------------
static void WaitForGpu() {
    // Signal the pending value for this slot, wait for it, then advance so
    // the stored value is again "next to signal".
    Check(gApp.queue->Signal(gApp.fence.Get(), gApp.fenceValues[gApp.frameIndex]), "Signal");

    if (gApp.fence->GetCompletedValue() < gApp.fenceValues[gApp.frameIndex]) {
        Check(gApp.fence->SetEventOnCompletion(gApp.fenceValues[gApp.frameIndex],
            gApp.fenceEvent), "SetEvent");
        WaitForSingleObject(gApp.fenceEvent, INFINITE);
    }

    // Confirmed: all prior GPU work complete. Advance for next use.
    gApp.fenceValues[gApp.frameIndex]++;
}

static void MoveToNextFrame() {
    // Schedule a signal marking completion of the frame just submitted.
    const UINT64 submitted = gApp.fenceValues[gApp.frameIndex];
    Check(gApp.queue->Signal(gApp.fence.Get(), submitted), "Signal");

    // Advance to the next swap-chain slot.
    gApp.frameIndex = gApp.swapChain->GetCurrentBackBufferIndex();

    // Confirm the PREVIOUS occupant of this slot has finished before we
    // recycle its command allocator / CB region / back buffer.
    if (gApp.fence->GetCompletedValue() < gApp.fenceValues[gApp.frameIndex]) {
        Check(gApp.fence->SetEventOnCompletion(gApp.fenceValues[gApp.frameIndex],
            gApp.fenceEvent), "SetEvent");
        WaitForSingleObject(gApp.fenceEvent, INFINITE);
    }

    // Reserve the next value for this slot's upcoming submission.
    gApp.fenceValues[gApp.frameIndex] = submitted + 1;
}

static ComPtr<ID3D12Resource> CreateDefaultTex2D(DXGI_FORMAT fmt, UINT w, UINT h,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initState) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    Check(gApp.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        initState, nullptr, IID_PPV_ARGS(&res)),
        "CreateDefaultTex2D");
    return res;
}

static ComPtr<ID3D12Resource> CreateUploadBuffer(UINT64 size) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> res;
    Check(gApp.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&res)),
        "CreateUploadBuffer");
    return res;
}

// Upload a tightly-packed CPU image into a DEFAULT texture via staging.
static void UploadTexture(ID3D12GraphicsCommandList* cl,
    ID3D12Resource* dst, DXGI_FORMAT fmt,
    const void* src, UINT w, UINT h, UINT texelBytes,
    ComPtr<ID3D12Resource>& outStaging) {
    const UINT rowPitch = (w * texelBytes + 255) & ~255u; // 256-byte row alignment
    const UINT64 totalSize = UINT64(rowPitch) * h;

    outStaging = CreateUploadBuffer(totalSize);

    uint8_t* mapped = nullptr;
    Check(outStaging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map staging");
    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (UINT row = 0; row < h; ++row)
        memcpy(mapped + row * rowPitch, s + row * w * texelBytes, size_t(w) * texelBytes);
    outStaging->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = dst;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = outStaging.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Footprint.Format = fmt;
    srcLoc.PlacedFootprint.Footprint.Width = w;
    srcLoc.PlacedFootprint.Footprint.Height = h;
    srcLoc.PlacedFootprint.Footprint.Depth = 1;
    srcLoc.PlacedFootprint.Footprint.RowPitch = rowPitch;

    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}

// ---------------------------------------------------------------------------
//  Size-dependent resources  (output UAV + swap-chain buffers)
// ---------------------------------------------------------------------------
void CreateSizeDependentResources() {
    // Output UAV texture
    gApp.outputTex = CreateDefaultTex2D(DXGI_FORMAT_R8G8B8A8_UNORM,
        gApp.width, gApp.height,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // UAV descriptor at heap slot 2
    D3D12_CPU_DESCRIPTOR_HANDLE h = gApp.srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += 2 * gApp.srvDescSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gApp.device->CreateUnorderedAccessView(gApp.outputTex.Get(), nullptr, &uav, h);

    // Back buffers
    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gApp.swapChain->GetBuffer(i, IID_PPV_ARGS(&gApp.backBuffers[i])), "GetBuffer");
}

void ResizeBackbuffer(int width, int height) {
    if (!gApp.device) { gApp.width = width; gApp.height = height; return; }

    WaitForGpu();

    for (UINT i = 0; i < FRAME_COUNT; ++i) {
        gApp.backBuffers[i].Reset();
        gApp.fenceValues[i] = gApp.fenceValues[gApp.frameIndex];
    }
    gApp.outputTex.Reset();

    gApp.width = std::max(1, width);
    gApp.height = std::max(1, height);

    Check(gApp.swapChain->ResizeBuffers(FRAME_COUNT, gApp.width, gApp.height,
        DXGI_FORMAT_R8G8B8A8_UNORM, 0),
        "ResizeBuffers");
    gApp.frameIndex = gApp.swapChain->GetCurrentBackBufferIndex();

    CreateSizeDependentResources();
}

// ---------------------------------------------------------------------------
//  One-time D3D12 initialisation
// ---------------------------------------------------------------------------
void InitD3D12(HWND hwnd) {
    gApp.hwnd = hwnd;

#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&gApp.device)),
        "D3D12CreateDevice");

    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Check(gApp.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&gApp.queue)), "CreateQueue");

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount = FRAME_COUNT;
    scd.Width = gApp.width;
    scd.Height = gApp.height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    ComPtr<IDXGISwapChain1> sc1;
    Check(factory->CreateSwapChainForHwnd(gApp.queue.Get(), hwnd, &scd, nullptr, nullptr, &sc1),
        "CreateSwapChain");
    Check(sc1.As(&gApp.swapChain), "QI SwapChain3");
    gApp.frameIndex = gApp.swapChain->GetCurrentBackBufferIndex();

    // Descriptor heap: [0]=height SRV, [1]=color SRV, [2]=output UAV
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.NumDescriptors = 3;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gApp.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gApp.srvHeap)), "CreateHeap");
    gApp.srvDescSize = gApp.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gApp.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&gApp.cmdAlloc[i])), "CreateAllocator");

    Check(gApp.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        gApp.cmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&gApp.cmdList)), "CreateCmdList");

    Check(gApp.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&gApp.fence)), "CreateFence");
    gApp.fenceValues[gApp.frameIndex] = 1;          // first value to signal
    gApp.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // --- Root signature: [0]=root CBV b0, [1]=table {t0,t1,u0} ---
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = params;

    ComPtr<ID3DBlob> sig, err;
    Check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err), "SerializeRS");
    Check(gApp.device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
        IID_PPV_ARGS(&gApp.rootSig)), "CreateRS");

    // --- Compile compute shader ---
    ComPtr<ID3DBlob> cs;
    Check(D3DCompile(g_shaderSrc, strlen(g_shaderSrc), nullptr, nullptr, nullptr,
        "CSMain", "cs_5_1", 0, 0, &cs, &err), "CompileCS");

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = gApp.rootSig.Get();
    psd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    Check(gApp.device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&gApp.pso)), "CreatePSO");

    // --- Constant buffer (UPLOAD, ring, persistently mapped) ---
    gApp.cbUpload = CreateUploadBuffer(FRAME_COUNT * CB_ALIGN);
    Check(gApp.cbUpload->Map(0, nullptr, reinterpret_cast<void**>(&gApp.cbMapped)), "Map CB");

    // --- Terrain textures (DEFAULT, immutable) ---
    gApp.heightTex = CreateDefaultTex2D(DXGI_FORMAT_R8_UINT, MAP_SIZE, MAP_SIZE,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    gApp.colorTex = CreateDefaultTex2D(DXGI_FORMAT_R32_UINT, MAP_SIZE, MAP_SIZE,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    ComPtr<ID3D12Resource> stagingH, stagingC;
    UploadTexture(gApp.cmdList.Get(), gApp.heightTex.Get(), DXGI_FORMAT_R8_UINT,
        gApp.heightMap.data(), MAP_SIZE, MAP_SIZE, 1, stagingH);
    UploadTexture(gApp.cmdList.Get(), gApp.colorTex.Get(), DXGI_FORMAT_R32_UINT,
        gApp.colorMap.data(), MAP_SIZE, MAP_SIZE, 4, stagingC);

    D3D12_RESOURCE_BARRIER toSrv[2] = {
        Transition(gApp.heightTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        Transition(gApp.colorTex.Get(),  D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
    };
    gApp.cmdList->ResourceBarrier(2, toSrv);
    Check(gApp.cmdList->Close(), "Close upload");
    ID3D12CommandList* lists[] = { gApp.cmdList.Get() };
    gApp.queue->ExecuteCommandLists(1, lists);
    WaitForGpu();
    // Staging buffers released here by ComPtr dtor. CPU colorMap no longer needed.
    gApp.colorMap.clear(); gApp.colorMap.shrink_to_fit();

    // --- SRVs ---
    D3D12_CPU_DESCRIPTOR_HANDLE h = gApp.srvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    srv.Format = DXGI_FORMAT_R8_UINT;
    gApp.device->CreateShaderResourceView(gApp.heightTex.Get(), &srv, h);
    h.ptr += gApp.srvDescSize;
    srv.Format = DXGI_FORMAT_R32_UINT;
    gApp.device->CreateShaderResourceView(gApp.colorTex.Get(), &srv, h);

    CreateSizeDependentResources();
}

// ---------------------------------------------------------------------------
//  Per-frame render  (CPU: command recording only)
// ---------------------------------------------------------------------------
void Render() {
    auto* alloc = gApp.cmdAlloc[gApp.frameIndex].Get();
    Check(alloc->Reset(), "Alloc Reset");
    Check(gApp.cmdList->Reset(alloc, gApp.pso.Get()), "List Reset");

    // Update ring constant buffer slot
    FrameConstants fc{ gApp.camX, gApp.camY, gApp.camZ, gApp.angle,
                       UINT(gApp.width), UINT(gApp.height), UINT(MAP_SIZE), 0 };
    memcpy(gApp.cbMapped + gApp.frameIndex * CB_ALIGN, &fc, sizeof(fc));

    gApp.cmdList->SetComputeRootSignature(gApp.rootSig.Get());
    ID3D12DescriptorHeap* heaps[] = { gApp.srvHeap.Get() };
    gApp.cmdList->SetDescriptorHeaps(1, heaps);
    gApp.cmdList->SetComputeRootConstantBufferView(0,
        gApp.cbUpload->GetGPUVirtualAddress() + gApp.frameIndex * CB_ALIGN);
    gApp.cmdList->SetComputeRootDescriptorTable(1,
        gApp.srvHeap->GetGPUDescriptorHandleForHeapStart());

    gApp.cmdList->Dispatch((gApp.width + 63) / 64, 1, 1);

    // output UAV -> COPY_SOURCE,  back buffer PRESENT -> COPY_DEST
    D3D12_RESOURCE_BARRIER pre[2] = {
        Transition(gApp.outputTex.Get(),
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        Transition(gApp.backBuffers[gApp.frameIndex].Get(),
                   D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)
    };
    gApp.cmdList->ResourceBarrier(2, pre);

    gApp.cmdList->CopyResource(gApp.backBuffers[gApp.frameIndex].Get(), gApp.outputTex.Get());

    D3D12_RESOURCE_BARRIER post[2] = {
        Transition(gApp.outputTex.Get(),
                   D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        Transition(gApp.backBuffers[gApp.frameIndex].Get(),
                   D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)
    };
    gApp.cmdList->ResourceBarrier(2, post);

    Check(gApp.cmdList->Close(), "Close");
    ID3D12CommandList* lists[] = { gApp.cmdList.Get() };
    gApp.queue->ExecuteCommandLists(1, lists);

    Check(gApp.swapChain->Present(1, 0), "Present");
    MoveToNextFrame();
}

// ---------------------------------------------------------------------------
//  Input / camera / window  (unchanged – CPU simulation)
// ---------------------------------------------------------------------------
bool KeyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

void WrapCamera() {
    while (gApp.camX < 0.0f)               gApp.camX += float(MAP_SIZE);
    while (gApp.camY < 0.0f)               gApp.camY += float(MAP_SIZE);
    while (gApp.camX >= float(MAP_SIZE))   gApp.camX -= float(MAP_SIZE);
    while (gApp.camY >= float(MAP_SIZE))   gApp.camY -= float(MAP_SIZE);
}

void UpdateCamera(float dt) {
    float moveSpeed = 90.0f, strafeSpeed = 75.0f, turnSpeed = 1.7f, liftSpeed = 65.0f;
    if (KeyDown(VK_SHIFT)) { moveSpeed *= 2; strafeSpeed *= 2; liftSpeed *= 2; }

    if (KeyDown(VK_LEFT) || KeyDown('A')) gApp.angle -= turnSpeed * dt;
    if (KeyDown(VK_RIGHT) || KeyDown('D')) gApp.angle += turnSpeed * dt;

    float fwd = 0, str = 0;
    if (KeyDown('W') || KeyDown(VK_UP))   fwd += 1;
    if (KeyDown('S') || KeyDown(VK_DOWN)) fwd -= 1;
    if (KeyDown('Q')) str -= 1;
    if (KeyDown('E')) str += 1;

    float sinA = std::sin(gApp.angle), cosA = std::cos(gApp.angle);
    gApp.camX += cosA * fwd * moveSpeed * dt;
    gApp.camY += sinA * fwd * moveSpeed * dt;
    gApp.camX += -sinA * str * strafeSpeed * dt;
    gApp.camY += cosA * str * strafeSpeed * dt;

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
    gApp.fpsTimer += dt; gApp.fpsFrames++;
    if (gApp.fpsTimer >= 0.5f) {
        float fps = float(gApp.fpsFrames) / gApp.fpsTimer;
        std::wstring t = L"Scratch Voxel Terrain Renderer | "
            L"W/S move, A/D turn, Q/E strafe, T fly on, G fly off, R/F vertical | FPS: "
            + std::to_wstring(int(fps));
        SetWindowTextW(hwnd, t.c_str());
        gApp.fpsTimer = 0; gApp.fpsFrames = 0;
    }
}

void ShutdownD3D12() {
    if (!gApp.device) return;

    // Drain the queue so no in-flight work references resources we're about
    // to release.
    WaitForGpu();

    if (gApp.cbMapped) { gApp.cbUpload->Unmap(0, nullptr); gApp.cbMapped = nullptr; }
    if (gApp.fenceEvent) { CloseHandle(gApp.fenceEvent); gApp.fenceEvent = nullptr; }
    // ComPtr destructors release the rest in declaration-reverse order.
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_SIZE: {
        int w = LOWORD(lParam), h = HIWORD(lParam);
        if (w > 0 && h > 0) ResizeBackbuffer(w, h);
        return 0;
    }
    case WM_CLOSE:   DestroyWindow(hwnd); return 0;
    case WM_DESTROY: gApp.running = false; PostQuitMessage(0); return 0;
    default:         return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}