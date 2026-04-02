#include "voxel.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstring>
#include <stdexcept>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
//  Global shared state
// ---------------------------------------------------------------------------
AppState gApp;

// ---------------------------------------------------------------------------
//  Internal constants
// ---------------------------------------------------------------------------
static constexpr UINT  FRAME_COUNT = 2;
static constexpr UINT  CB_ALIGN = 256;
static constexpr int   WATER_LEVEL = 50;

// ---------------------------------------------------------------------------
//  Frame constants – must match HLSL layout exactly
// ---------------------------------------------------------------------------
struct FrameConstants {
    float camPos[3];   float _p0;
    float fwd[3];      float _p1;
    float right[3];    float _p2;
    float up[3];       float _p3;
    uint32_t screenW, screenH, mapSize, mapHeight;
    float tanHalfFov, maxDist, _p4[2];
};
static_assert(sizeof(FrameConstants) <= CB_ALIGN, "CB overflow");

// ---------------------------------------------------------------------------
//  File-static D3D12 state
// ---------------------------------------------------------------------------
static struct GpuState {
    ComPtr<ID3D12Device>              device;
    ComPtr<ID3D12CommandQueue>        queue;
    ComPtr<IDXGISwapChain3>           swapChain;
    ComPtr<ID3D12CommandAllocator>    cmdAlloc[FRAME_COUNT];
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence>               fence;
    HANDLE                            fenceEvent = nullptr;
    UINT64                            fenceValues[FRAME_COUNT]{};
    UINT                              frameIndex = 0;

    ComPtr<ID3D12DescriptorHeap> srvHeap;
    UINT                         srvDescSize = 0;

    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> pso;

    ComPtr<ID3D12Resource> blockTex;            // Texture3D  R8_UINT
    ComPtr<ID3D12Resource> outputTex;           // Texture2D  UAV
    ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT];
    ComPtr<ID3D12Resource> cbUpload;
    uint8_t* cbMapped = nullptr;
} gpu;

// ---------------------------------------------------------------------------
//  Error handling
// ---------------------------------------------------------------------------
static void Check(HRESULT hr, const char* msg) {
    if (FAILED(hr)) { OutputDebugStringA(msg); throw std::runtime_error(msg); }
}

// ---------------------------------------------------------------------------
//  Math / noise  (CPU-side terrain authoring)
// ---------------------------------------------------------------------------
static float Lerp(float a, float b, float t) { return a + (b - a) * t; }
static float Smooth(float t) { return t * t * (3.0f - 2.0f * t); }

static uint32_t Hash2D(int x, int y) {
    uint32_t h = 2166136261u;
    h = (h ^ uint32_t(x)) * 16777619u;
    h = (h ^ uint32_t(y)) * 16777619u;
    h ^= (h >> 13); h *= 1274126177u; h ^= (h >> 16);
    return h;
}
static float Random01(int x, int y) {
    return float(Hash2D(x, y) & 0x00FFFFFF) / float(0x00FFFFFF);
}

float ValueNoise(float x, float y) {
    int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    float sx = Smooth(x - float(x0)), sy = Smooth(y - float(y0));
    float a = Lerp(Random01(x0, y0), Random01(x0 + 1, y0), sx);
    float b = Lerp(Random01(x0, y0 + 1), Random01(x0 + 1, y0 + 1), sx);
    return Lerp(a, b, sy);
}

float FBM(float x, float y, int octaves) {
    float sum = 0, amp = 0.5f, freq = 1, norm = 0;
    for (int i = 0; i < octaves; ++i) {
        sum += ValueNoise(x * freq, y * freq) * amp;
        norm += amp; amp *= 0.5f; freq *= 2.0f;
    }
    return sum / norm;
}

static int WrapCoord(int v) { return v & (MAP_SIZE - 1); }

static uint8_t SampleHeightNearest(float wx, float wz) {
    return gApp.heightMap[WrapCoord(int(std::floor(wz))) * MAP_SIZE
        + WrapCoord(int(std::floor(wx)))];
}

// ---------------------------------------------------------------------------
//  Terrain generation  (fills 3-D blockMap + 2-D heightMap cache)
// ---------------------------------------------------------------------------
void GenerateTerrain() {
    const size_t totalBlocks = size_t(MAP_SIZE) * MAP_HEIGHT * MAP_SIZE;
    gApp.blockMap.assign(totalBlocks, BLOCK_AIR);
    gApp.heightMap.resize(size_t(MAP_SIZE) * MAP_SIZE);

    for (int z = 0; z < MAP_SIZE; ++z) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            float nx = x * 0.0035f, nz = z * 0.0035f;
            float large = FBM(x * 0.0012f, z * 0.0012f, 5);
            float medium = FBM(nx, nz, 6);
            float detail = FBM(x * 0.012f, z * 0.012f, 4);

            float e = 0.55f * medium + 0.30f * large + 0.15f * detail;
            e = std::pow(e, 1.35f);

            float ridge = std::fabs(FBM(x * 0.006f, z * 0.006f, 5) - 0.5f) * 2.0f;
            float heightF = 20.0f + e * 170.0f + ridge * 18.0f;
            int   surfH = std::clamp(int(heightF), 0, MAP_HEIGHT - 1);

            gApp.heightMap[z * MAP_SIZE + x] = static_cast<uint8_t>(surfH);

            // Determine surface block type from the same height-based palette
            BlockType surfType;
            if (surfH < 42)  surfType = BLOCK_SAND;
            else if (surfH < 52)  surfType = BLOCK_SAND;
            else if (surfH < 58)  surfType = BLOCK_SAND;
            else if (surfH < 140) surfType = BLOCK_GRASS;
            else if (surfH < 180) surfType = BLOCK_ROCK;
            else                  surfType = BLOCK_SNOW;

            // Fill column
            for (int y = 0; y <= surfH; ++y) {
                BlockType bt;
                if (y < surfH - 4) bt = BLOCK_STONE;
                else if (y < surfH)     bt = BLOCK_DIRT;
                else                    bt = surfType;
                gApp.blockMap[BlockIndex(WrapCoord(x), y, WrapCoord(z))] = bt;
            }

            // Water fill above solid ground up to water level
            for (int y = surfH + 1; y <= WATER_LEVEL && y < MAP_HEIGHT; ++y) {
                gApp.blockMap[BlockIndex(WrapCoord(x), y, WrapCoord(z))] =
                    (y < 42) ? BLOCK_DEEP_WATER : BLOCK_WATER;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  HLSL compute shader — 3-D DDA voxel raycaster
// ---------------------------------------------------------------------------
static const char* g_shaderSrc = R"(
cbuffer FrameConstants : register(b0)
{
    float3 camPos;    float _p0;
    float3 camFwd;    float _p1;
    float3 camRight;  float _p2;
    float3 camUp;     float _p3;
    uint   screenW, screenH, mapSize, mapHeight;
    float  tanHalfFov, maxDist;
    float2 _p4;
};

Texture3D<uint>     BlockMap : register(t0);
RWTexture2D<float4> Output   : register(u0);

static const int   MAX_STEPS = 512;
static const float3 FOG_COLOR = float3(0.67, 0.80, 0.92);

// ---- block colours (linear-ish RGB) ----
static const float3 bcolors[9] = {
    float3(0,0,0),                // 0 air
    float3(0.50, 0.50, 0.50),    // 1 stone
    float3(0.55, 0.36, 0.18),    // 2 dirt
    float3(0.30, 0.58, 0.16),    // 3 grass (top)
    float3(0.76, 0.70, 0.50),    // 4 sand
    float3(0.15, 0.40, 0.68),    // 5 water
    float3(0.91, 0.91, 0.96),    // 6 snow
    float3(0.05, 0.14, 0.38),    // 7 deep water
    float3(0.42, 0.42, 0.39),    // 8 rock
};

// face brightness:  +X  -X  +Y(top)  -Y(bottom)  +Z  -Z
static const float faceBright[6] = { 0.80, 0.80, 1.00, 0.45, 0.65, 0.65 };

uint WrapXZ(int v) { return uint(v) & (mapSize - 1u); }

uint GetBlock(int3 p)
{
    if (p.y < 0 || p.y >= (int)mapHeight) return 0u;
    return BlockMap.Load(int4(WrapXZ(p.x), p.y, WrapXZ(p.z), 0));
}

uint SimpleHash(int3 p)
{
    uint h = uint(p.x) * 374761393u + uint(p.y) * 668265263u + uint(p.z) * 1274126177u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return h;
}

float3 BlockColor(uint bt, int faceIdx, int3 bp)
{
    float3 c = bcolors[bt];
    // grass: green on top, dirt on sides / bottom
    if (bt == 3 && faceIdx != 2) c = bcolors[2];
    // snow: slightly blue on sides
    if (bt == 6 && faceIdx != 2) c *= float3(0.90, 0.92, 0.98);

    // subtle per-block variation
    float v = float(SimpleHash(bp) & 255u) / 255.0 * 0.08 - 0.04;
    c *= (1.0 + v);
    return c;
}

[numthreads(8,8,1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= screenW || tid.y >= screenH) return;

    float aspect = float(screenW) / float(max(1u, screenH));
    float px = (2.0 * (float(tid.x) + 0.5) / float(screenW) - 1.0) * aspect * tanHalfFov;
    float py = (1.0 - 2.0 * (float(tid.y) + 0.5) / float(screenH)) * tanHalfFov;

    float3 rd = normalize(camFwd + camRight * px + camUp * py);

    // ---- 3-D DDA setup ----
    int3  mapPos = int3(floor(camPos));
    int3  step   = int3(rd.x >= 0 ? 1 : -1,
                        rd.y >= 0 ? 1 : -1,
                        rd.z >= 0 ? 1 : -1);
    float3 tDelta = abs(1.0 / rd);
    float3 tMax;
    tMax.x = ((rd.x >= 0) ? (float(mapPos.x + 1) - camPos.x)
                           : (camPos.x - float(mapPos.x))) * tDelta.x;
    tMax.y = ((rd.y >= 0) ? (float(mapPos.y + 1) - camPos.y)
                           : (camPos.y - float(mapPos.y))) * tDelta.y;
    tMax.z = ((rd.z >= 0) ? (float(mapPos.z + 1) - camPos.z)
                           : (camPos.z - float(mapPos.z))) * tDelta.z;

    float dist     = 0;
    int   faceIdx  = -1;   // 0:+X  1:-X  2:+Y  3:-Y  4:+Z  5:-Z
    bool  hit      = false;

    [loop]
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        // advance to next voxel boundary
        if (tMax.x < tMax.y)
        {
            if (tMax.x < tMax.z)
            { dist = tMax.x; tMax.x += tDelta.x; mapPos.x += step.x;
              faceIdx = step.x > 0 ? 1 : 0; }
            else
            { dist = tMax.z; tMax.z += tDelta.z; mapPos.z += step.z;
              faceIdx = step.z > 0 ? 5 : 4; }
        }
        else
        {
            if (tMax.y < tMax.z)
            { dist = tMax.y; tMax.y += tDelta.y; mapPos.y += step.y;
              faceIdx = step.y > 0 ? 3 : 2; }
            else
            { dist = tMax.z; tMax.z += tDelta.z; mapPos.z += step.z;
              faceIdx = step.z > 0 ? 5 : 4; }
        }

        if (dist > maxDist) break;
        if (mapPos.y < 0) break;
        if (mapPos.y >= (int)mapHeight) continue;

        uint bt = GetBlock(mapPos);
        if (bt != 0u) { hit = true; break; }
    }

    float4 color;
    if (hit)
    {
        uint bt = GetBlock(mapPos);
        float3 c = BlockColor(bt, faceIdx, mapPos);
        c *= faceBright[faceIdx];

        float fogT = saturate(dist / maxDist);
        fogT *= fogT;
        c = lerp(c, FOG_COLOR, fogT);
        color = float4(c, 1.0);
    }
    else
    {
        // sky gradient based on ray pitch
        float t = rd.y * 0.5 + 0.5;
        float3 lo = float3(0.67, 0.80, 0.92);
        float3 hi = float3(0.25, 0.45, 0.75);
        color = float4(lerp(lo, hi, saturate(t)), 1.0);
    }

    Output[tid.xy] = color;
}
)";

// ---------------------------------------------------------------------------
//  D3D12 helpers
// ---------------------------------------------------------------------------
static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

static void WaitForGpu() {
    Check(gpu.queue->Signal(gpu.fence.Get(), gpu.fenceValues[gpu.frameIndex]), "Signal");
    if (gpu.fence->GetCompletedValue() < gpu.fenceValues[gpu.frameIndex]) {
        Check(gpu.fence->SetEventOnCompletion(gpu.fenceValues[gpu.frameIndex],
            gpu.fenceEvent), "SetEvent");
        WaitForSingleObject(gpu.fenceEvent, INFINITE);
    }
    gpu.fenceValues[gpu.frameIndex]++;
}

static void MoveToNextFrame() {
    const UINT64 submitted = gpu.fenceValues[gpu.frameIndex];
    Check(gpu.queue->Signal(gpu.fence.Get(), submitted), "Signal");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();
    if (gpu.fence->GetCompletedValue() < gpu.fenceValues[gpu.frameIndex]) {
        Check(gpu.fence->SetEventOnCompletion(gpu.fenceValues[gpu.frameIndex],
            gpu.fenceEvent), "SetEvent");
        WaitForSingleObject(gpu.fenceEvent, INFINITE);
    }
    gpu.fenceValues[gpu.frameIndex] = submitted + 1;
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
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&res)),
        "CreateUploadBuffer");
    return res;
}

static ComPtr<ID3D12Resource> CreateDefaultTex2D(DXGI_FORMAT fmt, UINT w, UINT h,
    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w; rd.Height = h;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        state, nullptr, IID_PPV_ARGS(&res)), "CreateTex2D");
    return res;
}

static ComPtr<ID3D12Resource> CreateDefaultTex3D(DXGI_FORMAT fmt,
    UINT w, UINT h, UINT d,
    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    rd.Width = w; rd.Height = h;
    rd.DepthOrArraySize = static_cast<UINT16>(d);
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    Check(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        state, nullptr, IID_PPV_ARGS(&res)), "CreateTex3D");
    return res;
}

// Upload tightly-packed CPU data into a DEFAULT Texture3D via staging.
static void UploadTexture3D(ID3D12GraphicsCommandList* cl,
    ID3D12Resource* dst, DXGI_FORMAT fmt,
    const void* src, UINT w, UINT h, UINT d, UINT texelBytes,
    ComPtr<ID3D12Resource>& outStaging) {

    const UINT   rowPitch = (w * texelBytes + 255u) & ~255u;
    const UINT64 slicePitch = UINT64(rowPitch) * h;
    const UINT64 totalSize = slicePitch * d;

    outStaging = CreateUploadBuffer(totalSize);

    uint8_t* mapped = nullptr;
    Check(outStaging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map3D");
    const uint8_t* s = static_cast<const uint8_t*>(src);
    const UINT srcRow = w * texelBytes;
    for (UINT z = 0; z < d; ++z)
        for (UINT y = 0; y < h; ++y)
            memcpy(mapped + z * slicePitch + y * rowPitch,
                s + (size_t(z) * h + y) * srcRow,
                srcRow);
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
    srcLoc.PlacedFootprint.Footprint.Depth = d;
    srcLoc.PlacedFootprint.Footprint.RowPitch = rowPitch;

    cl->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}

// ---------------------------------------------------------------------------
//  Size-dependent resources
// ---------------------------------------------------------------------------
static void CreateSizeDependentResources() {
    gpu.outputTex = CreateDefaultTex2D(DXGI_FORMAT_R8G8B8A8_UNORM,
        gApp.width, gApp.height,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // UAV at heap slot 1
    D3D12_CPU_DESCRIPTOR_HANDLE h = gpu.srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += 1 * gpu.srvDescSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(gpu.outputTex.Get(), nullptr, &uav, h);

    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gpu.swapChain->GetBuffer(i, IID_PPV_ARGS(&gpu.backBuffers[i])), "GetBuffer");
}

void ResizeBackbuffer(int width, int height) {
    if (!gpu.device) { gApp.width = width; gApp.height = height; return; }
    WaitForGpu();
    for (UINT i = 0; i < FRAME_COUNT; ++i) {
        gpu.backBuffers[i].Reset();
        gpu.fenceValues[i] = gpu.fenceValues[gpu.frameIndex];
    }
    gpu.outputTex.Reset();
    gApp.width = std::max(1, width);
    gApp.height = std::max(1, height);
    Check(gpu.swapChain->ResizeBuffers(FRAME_COUNT, gApp.width, gApp.height,
        DXGI_FORMAT_R8G8B8A8_UNORM, 0), "Resize");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();
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
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Factory");
    Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&gpu.device)), "Device");

    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Check(gpu.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&gpu.queue)), "Queue");

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount = FRAME_COUNT;
    scd.Width = gApp.width;
    scd.Height = gApp.height;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.SampleDesc.Count = 1;
    ComPtr<IDXGISwapChain1> sc1;
    Check(factory->CreateSwapChainForHwnd(gpu.queue.Get(), hwnd, &scd,
        nullptr, nullptr, &sc1), "SwapChain");
    Check(sc1.As(&gpu.swapChain), "QI SC3");
    gpu.frameIndex = gpu.swapChain->GetCurrentBackBufferIndex();

    // Descriptor heap: [0]=block SRV, [1]=output UAV
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.NumDescriptors = 2;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&gpu.srvHeap)), "Heap");
    gpu.srvDescSize = gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (UINT i = 0; i < FRAME_COUNT; ++i)
        Check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&gpu.cmdAlloc[i])), "Alloc");

    Check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        gpu.cmdAlloc[0].Get(), nullptr, IID_PPV_ARGS(&gpu.cmdList)), "CmdList");

    Check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&gpu.fence)), "Fence");
    gpu.fenceValues[gpu.frameIndex] = 1;
    gpu.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // ---- Root signature: [0] root CBV b0, [1] table {t0, u0} ----
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

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
    Check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
        &sig, &err), "SerializeRS");
    Check(gpu.device->CreateRootSignature(0, sig->GetBufferPointer(),
        sig->GetBufferSize(), IID_PPV_ARGS(&gpu.rootSig)), "RS");

    // ---- Compile compute shader ----
    ComPtr<ID3DBlob> cs;
    HRESULT hr = D3DCompile(g_shaderSrc, strlen(g_shaderSrc), nullptr, nullptr, nullptr,
        "CSMain", "cs_5_1", 0, 0, &cs, &err);
    if (FAILED(hr)) {
        if (err) OutputDebugStringA(static_cast<const char*>(err->GetBufferPointer()));
        Check(hr, "CompileCS");
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = gpu.rootSig.Get();
    psd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    Check(gpu.device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&gpu.pso)), "PSO");

    // ---- Constant buffer ring ----
    gpu.cbUpload = CreateUploadBuffer(FRAME_COUNT * CB_ALIGN);
    Check(gpu.cbUpload->Map(0, nullptr, reinterpret_cast<void**>(&gpu.cbMapped)), "MapCB");

    // ---- Block texture (Texture3D R8_UINT) ----
    gpu.blockTex = CreateDefaultTex3D(DXGI_FORMAT_R8_UINT,
        MAP_SIZE, MAP_HEIGHT, MAP_SIZE,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    ComPtr<ID3D12Resource> staging;
    UploadTexture3D(gpu.cmdList.Get(), gpu.blockTex.Get(), DXGI_FORMAT_R8_UINT,
        gApp.blockMap.data(), MAP_SIZE, MAP_HEIGHT, MAP_SIZE, 1, staging);

    auto bar = Transition(gpu.blockTex.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.cmdList->ResourceBarrier(1, &bar);

    Check(gpu.cmdList->Close(), "Close upload");
    ID3D12CommandList* lists[] = { gpu.cmdList.Get() };
    gpu.queue->ExecuteCommandLists(1, lists);
    WaitForGpu();
    // staging released here; blockMap kept for future CPU-side block edits

    // ---- SRV for block texture (heap slot 0) ----
    D3D12_CPU_DESCRIPTOR_HANDLE srvH = gpu.srvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8_UINT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture3D.MipLevels = 1;
    gpu.device->CreateShaderResourceView(gpu.blockTex.Get(), &srv, srvH);

    CreateSizeDependentResources();
}

// ---------------------------------------------------------------------------
//  Per-frame render
// ---------------------------------------------------------------------------
void Render() {
    auto* alloc = gpu.cmdAlloc[gpu.frameIndex].Get();
    Check(alloc->Reset(), "AllocReset");
    Check(gpu.cmdList->Reset(alloc, gpu.pso.Get()), "ListReset");

    // Build camera basis on CPU
    float cosP = std::cos(gApp.pitch), sinP = std::sin(gApp.pitch);
    float cosA = std::cos(gApp.angle), sinA = std::sin(gApp.angle);

    FrameConstants fc{};
    // World coords: X = gApp.camX,  Y(up) = gApp.camZ,  Z = gApp.camY
    fc.camPos[0] = gApp.camX;
    fc.camPos[1] = gApp.camZ;
    fc.camPos[2] = gApp.camY;
    fc.fwd[0] = cosP * cosA;
    fc.fwd[1] = sinP;
    fc.fwd[2] = cosP * sinA;
    fc.right[0] = -sinA;
    fc.right[1] = 0.0f;
    fc.right[2] = cosA;
    // up = cross(right, forward)
    fc.up[0] = -sinP * cosA;
    fc.up[1] = cosP;
    fc.up[2] = -sinP * sinA;
    fc.screenW = UINT(gApp.width);
    fc.screenH = UINT(gApp.height);
    fc.mapSize = UINT(MAP_SIZE);
    fc.mapHeight = UINT(MAP_HEIGHT);
    fc.tanHalfFov = 0.8f;
    fc.maxDist = 300.0f;

    memcpy(gpu.cbMapped + gpu.frameIndex * CB_ALIGN, &fc, sizeof(fc));

    gpu.cmdList->SetComputeRootSignature(gpu.rootSig.Get());
    ID3D12DescriptorHeap* heaps[] = { gpu.srvHeap.Get() };
    gpu.cmdList->SetDescriptorHeaps(1, heaps);
    gpu.cmdList->SetComputeRootConstantBufferView(0,
        gpu.cbUpload->GetGPUVirtualAddress() + gpu.frameIndex * CB_ALIGN);
    gpu.cmdList->SetComputeRootDescriptorTable(1,
        gpu.srvHeap->GetGPUDescriptorHandleForHeapStart());

    gpu.cmdList->Dispatch((gApp.width + 7) / 8, (gApp.height + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER pre[2] = {
        Transition(gpu.outputTex.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        Transition(gpu.backBuffers[gpu.frameIndex].Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)
    };
    gpu.cmdList->ResourceBarrier(2, pre);
    gpu.cmdList->CopyResource(gpu.backBuffers[gpu.frameIndex].Get(), gpu.outputTex.Get());

    D3D12_RESOURCE_BARRIER post[2] = {
        Transition(gpu.outputTex.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        Transition(gpu.backBuffers[gpu.frameIndex].Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)
    };
    gpu.cmdList->ResourceBarrier(2, post);

    Check(gpu.cmdList->Close(), "Close");
    ID3D12CommandList* lists[] = { gpu.cmdList.Get() };
    gpu.queue->ExecuteCommandLists(1, lists);
    Check(gpu.swapChain->Present(1, 0), "Present");
    MoveToNextFrame();
}

// ---------------------------------------------------------------------------
//  Shutdown
// ---------------------------------------------------------------------------
void ShutdownD3D12() {
    if (!gpu.device) return;
    WaitForGpu();
    if (gpu.cbMapped) { gpu.cbUpload->Unmap(0, nullptr); gpu.cbMapped = nullptr; }
    if (gpu.fenceEvent) { CloseHandle(gpu.fenceEvent); gpu.fenceEvent = nullptr; }
}

// ---------------------------------------------------------------------------
//  Input / camera
// ---------------------------------------------------------------------------
static bool KeyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

static void WrapCamera() {
    while (gApp.camX < 0.0f)             gApp.camX += float(MAP_SIZE);
    while (gApp.camY < 0.0f)             gApp.camY += float(MAP_SIZE);
    while (gApp.camX >= float(MAP_SIZE))  gApp.camX -= float(MAP_SIZE);
    while (gApp.camY >= float(MAP_SIZE))  gApp.camY -= float(MAP_SIZE);
}

static void CenterCursor() {
    RECT r; GetClientRect(gApp.hwnd, &r);
    POINT c = { (r.right - r.left) / 2, (r.bottom - r.top) / 2 };
    ClientToScreen(gApp.hwnd, &c);
    SetCursorPos(c.x, c.y);
}

void UpdateCamera(float dt) {
    float moveSpeed = 8.0f, strafeSpeed = 7.0f, liftSpeed = 8.0f;
    if (KeyDown(VK_SHIFT)) { moveSpeed *= 3; strafeSpeed *= 3; liftSpeed *= 3; }

    // ---- mouse look ----
    if (gApp.mouseCaptured) {
        RECT cr; GetClientRect(gApp.hwnd, &cr);
        POINT center = { (cr.right - cr.left) / 2, (cr.bottom - cr.top) / 2 };
        POINT screenCenter = center;
        ClientToScreen(gApp.hwnd, &screenCenter);

        POINT cur; GetCursorPos(&cur);
        float dx = float(cur.x - screenCenter.x);
        float dy = float(cur.y - screenCenter.y);
        const float sens = 0.002f;
        gApp.angle += dx * sens;
        gApp.pitch -= dy * sens;
        gApp.pitch = std::clamp(gApp.pitch, -1.48f, 1.48f);  // ~±85°
        SetCursorPos(screenCenter.x, screenCenter.y);
    }

    // ---- keyboard look (fallback) ----
    if (KeyDown(VK_LEFT))  gApp.angle -= 1.7f * dt;
    if (KeyDown(VK_RIGHT)) gApp.angle += 1.7f * dt;

    // ---- movement (horizontal, independent of pitch) ----
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

    // ---- vertical / fly ----
    if (KeyDown('T')) gApp.flyMode = true;
    if (KeyDown('G')) gApp.flyMode = false;

    if (gApp.flyMode) {
        if (KeyDown(VK_SPACE) || KeyDown('R')) gApp.camZ += liftSpeed * dt;
        if (KeyDown(VK_CONTROL) || KeyDown('F')) gApp.camZ -= liftSpeed * dt;
    }
    else {
        float surfH = float(SampleHeightNearest(gApp.camX, gApp.camY));
        float eyeTarget = surfH + 1.0f + 1.5f;   // top-of-block + eye offset
        float follow = std::clamp(dt * 8.0f, 0.0f, 1.0f);
        gApp.camZ = Lerp(gApp.camZ, eyeTarget, follow);
    }
    WrapCamera();
}

void UpdateWindowTitle(HWND hwnd, float dt) {
    gApp.fpsTimer += dt; gApp.fpsFrames++;
    if (gApp.fpsTimer >= 0.5f) {
        float fps = float(gApp.fpsFrames) / gApp.fpsTimer;
        std::wstring t = L"Voxel Block Terrain | "
            L"W/S/A/D move, mouse look, T fly, G walk, Space/Ctrl up/down | FPS: "
            + std::to_wstring(int(fps));
        SetWindowTextW(hwnd, t.c_str());
        gApp.fpsTimer = 0; gApp.fpsFrames = 0;
    }
}

// ---------------------------------------------------------------------------
//  Window procedure
// ---------------------------------------------------------------------------
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