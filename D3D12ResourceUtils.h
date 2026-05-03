// D3D12ResourceUtils.h

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

void Check(HRESULT hr, const char* message);

D3D12_RESOURCE_BARRIER Transition(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after
);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateUploadBuffer(
    ID3D12Device* device,
    UINT64 size
);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultTex2D(
    ID3D12Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState
);

Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultTex3D(
    ID3D12Device* device,
    DXGI_FORMAT format,
    UINT width,
    UINT height,
    UINT depth,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState
);