#include "D3D12ResourceUtils.h"

#include <stdexcept>


// ===================================================================
//  Error helper
// ===================================================================
void Check(HRESULT hr, const char* message) {
    if (FAILED(hr)) {
        OutputDebugStringA(message);
        throw std::runtime_error(message);
    }
}

// ===================================================================
//  D3D12 utilities
// ===================================================================
D3D12_RESOURCE_BARRIER Transition(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after
) {
    D3D12_RESOURCE_BARRIER barrier{};

    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;

    return barrier;
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateUploadBuffer(
    ID3D12Device* device,
    UINT64 size
) {
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;

    Check(
        device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&resource)
        ),
        "UploadBuf"
    );

    return resource;
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultTex2D(
    ID3D12Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState
) {
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = format;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Flags = flags;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;

    Check(
        device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            initialState,
            nullptr,
            IID_PPV_ARGS(&resource)
        ),
        "Tex2D"
    );

    return resource;
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultTex3D(
    ID3D12Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    UINT depth,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState
) {
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = UINT16(depth);
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = format;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Flags = flags;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;

    Check(
        device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            initialState,
            nullptr,
            IID_PPV_ARGS(&resource)
        ),
        "Tex3D"
    );

    return resource;
}