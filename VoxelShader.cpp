#include "VoxelShader.h"

// ===================================================================
//  HLSL compute shader – two-level (section / voxel) DDA through
//  toroidal atlas, using a section-occupancy map to skip empty
//  32×16×32 regions in a single step.
// ===================================================================
const char* g_shaderSrc = R"(
cbuffer CB : register(b0)
{
    float3 camPos;   float _p0;
    float3 camFwd;   float _p1;
    float3 camRight; float _p2;
    float3 camUp;    float _p3;
    uint   screenW, screenH;
    int    loadMinCX, loadMinCZ;
    int    loadMaxCX, loadMaxCZ;
    float  tanHalfFov, maxDist;
};

Texture3D<uint>     Atlas     : register(t0);
Texture3D<uint>     Occupancy : register(t1);
RWTexture2D<float4> Output    : register(u0);

static const int   LC   = 24;
static const int   CX   = 32;
static const int   CY   = 128;
static const int   SY   = 16;
static const int   SPC  = 8;

static const int3   SEC_I = int3(CX, SY, CX);
static const float3 SEC_F = float3(32.0, 16.0, 32.0);

static const int   MAX_COARSE = 64;
static const int   MAX_FINE   = 96;

static const float3 FOG_CLR = float3(0.67, 0.80, 0.92);

static const float3 bcolors[9] = {
    float3(0,0,0),
    float3(0.50,0.50,0.50),
    float3(0.55,0.36,0.18),
    float3(0.30,0.58,0.16),
    float3(0.76,0.70,0.50),
    float3(0.15,0.40,0.68),
    float3(0.91,0.91,0.96),
    float3(0.05,0.14,0.38),
    float3(0.42,0.42,0.39),
};
static const float faceBri[6] = { 0.80, 0.80, 1.00, 0.45, 0.65, 0.65 };

// Outward surface normal per hit-face index.
static const float3 faceN[6] = {
    float3( 1, 0, 0), float3(-1, 0, 0),
    float3( 0, 1, 0), float3( 0,-1, 0),
    float3( 0, 0, 1), float3( 0, 0,-1)
};
// Deterministic axis-aligned tangent / bitangent per face.
static const float3 faceT[6] = {
    float3( 0, 0, 1), float3( 0, 0, 1),
    float3( 1, 0, 0), float3( 1, 0, 0),
    float3( 1, 0, 0), float3( 1, 0, 0)
};
static const float3 faceB[6] = {
    float3( 0, 1, 0), float3( 0, 1, 0),
    float3( 0, 0, 1), float3( 0, 0, 1),
    float3( 0, 1, 0), float3( 0, 1, 0)
};

// ---- Ambient-occlusion parameters ----
static const int   AO_SAMPLES  = 6;
static const float AO_MAX_DIST = 12.0;
static const float AO_BIAS     = 0.02;
static const float AO_STRENGTH = 0.9;

// Directional influence: sun-facing surfaces receive less AO shadowing.
// Set AO_SUN_INFLUENCE to 0.0 for omnidirectional AO (original behaviour).
static const float3 AO_SUN_DIR       = float3(0.8305, 0.4983, 0.2491); // morning sun high in east
static const float  AO_SUN_INFLUENCE = 1.0;   // 0 = omni, 1 = full directional
static const float  AO_MIN_BRIGHT    = 0.12;  // floor to prevent pure-black backfaces

// 6 fixed hemisphere directions in tangent space (z = along normal).
static const float3 AO_DIRS[6] = {
    float3( 0.87543,  0.23457, 0.42262),
    float3(-0.64085,  0.64085, 0.42262),
    float3(-0.23457, -0.87543, 0.42262),
    float3( 0.12941,  0.48296, 0.86603),
    float3(-0.48296, -0.12941, 0.86603),
    float3( 0.35355, -0.35355, 0.86603)
};

uint PosMod(int v, int m) { return uint(((v % m) + m) % m); }

int3 SectionOf(int3 p) { return int3(p.x >> 5, p.y >> 4, p.z >> 5); }

float3 InitTMax(int3 cell, float3 cellSize, float3 origin, int3 st, float3 tD)
{
    float3 lo = float3(cell) * cellSize;
    float3 hi = lo + cellSize;
    float3 d;
    d.x = (st.x > 0) ? (hi.x - origin.x) : (origin.x - lo.x);
    d.y = (st.y > 0) ? (hi.y - origin.y) : (origin.y - lo.y);
    d.z = (st.z > 0) ? (hi.z - origin.z) : (origin.z - lo.z);
    return d * tD;
}

uint GetBlock(int3 wp)
{
    if (wp.y < 0 || wp.y >= CY) return 0u;
    int cx = wp.x >> 5;
    int cz = wp.z >> 5;
    if (cx < loadMinCX || cx >= loadMaxCX ||
        cz < loadMinCZ || cz >= loadMaxCZ) return 0u;
    uint sx = PosMod(cx, LC);
    uint sz = PosMod(cz, LC);
    uint lx = uint(wp.x) & 31u;
    uint lz = uint(wp.z) & 31u;
    return Atlas.Load(int4(sx * CX + lx, wp.y, sz * CX + lz, 0));
}

uint GetOccupancy(int3 sc)
{
    if (sc.y < 0 || sc.y >= SPC) return 0u;
    int cx = sc.x;
    int cz = sc.z;
    if (cx < loadMinCX || cx >= loadMaxCX ||
        cz < loadMinCZ || cz >= loadMaxCZ) return 0u;
    uint sx = PosMod(cx, LC);
    uint sz = PosMod(cz, LC);
    return Occupancy.Load(int4(sx, sc.y, sz, 0));
}

uint BHash(int3 p)
{
    uint h = uint(p.x)*374761393u + uint(p.y)*668265263u + uint(p.z)*1274126177u;
    h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
    return h;
}

float3 BlockColor(uint bt, int face, int3 bp)
{
    float3 c = bcolors[bt];
    if (bt == 3 && face != 2) c = bcolors[2];
    if (bt == 6 && face != 2) c *= float3(0.90,0.92,0.98);
    float v = float(BHash(bp) & 255u) / 255.0 * 0.08 - 0.04;
    return c * (1.0 + v);
}

bool Trace(float3 ro, float3 rd, float maxT,
           out float outDist, out int outFace, out int3 outVox)
{
    int3   st = int3(rd.x>=0?1:-1, rd.y>=0?1:-1, rd.z>=0?1:-1);
    float3 tD = abs(1.0/rd);

    int3   mp = int3(floor(ro));
    float3 tM = InitTMax(mp, float3(1,1,1), ro, st, tD);

    int3   sc  = SectionOf(mp);
    float3 tDC = SEC_F * tD;
    float3 tMC = InitTMax(sc, SEC_F, ro, st, tD);

    float dist = 0.0;
    int   face = -1;
    bool  hit  = false;
    bool  fineValid = true;
    bool  skipFirst = true;

    [loop] for (int ci = 0; ci < MAX_COARSE; ++ci)
    {
        if (dist > maxT) break;
        if (sc.y <  0   && st.y < 0) break;
        if (sc.y >= SPC && st.y > 0) break;

        if (GetOccupancy(sc) != 0u)
        {
            int3 lo = sc * SEC_I;
            int3 hi = lo + SEC_I - int3(1,1,1);

            if (!fineValid) {
                float3 p = ro + rd * dist;
                mp = clamp(int3(floor(p)), lo, hi);
                tM = InitTMax(mp, float3(1,1,1), ro, st, tD);
                fineValid = true;
            }

            [loop] for (int fi = 0; fi < MAX_FINE; ++fi)
            {
                if (!skipFirst) {
                    uint bt = GetBlock(mp);
                    if (bt != 0u) { hit = true; break; }
                }
                skipFirst = false;

                if (tM.x < tM.y) {
                    if (tM.x < tM.z) { dist=tM.x; tM.x+=tD.x; mp.x+=st.x; face=st.x>0?1:0; }
                    else             { dist=tM.z; tM.z+=tD.z; mp.z+=st.z; face=st.z>0?5:4; }
                } else {
                    if (tM.y < tM.z) { dist=tM.y; tM.y+=tD.y; mp.y+=st.y; face=st.y>0?3:2; }
                    else             { dist=tM.z; tM.z+=tD.z; mp.z+=st.z; face=st.z>0?5:4; }
                }

                if (dist > maxT) break;
                if (any(mp < lo) || any(mp > hi)) break;
            }
            if (hit || dist > maxT) break;

            sc  = SectionOf(mp);
            tMC = InitTMax(sc, SEC_F, ro, st, tD);
        }
        else
        {
            skipFirst = false;
            if (tMC.x < tMC.y) {
                if (tMC.x < tMC.z) { dist=tMC.x; tMC.x+=tDC.x; sc.x+=st.x; face=st.x>0?1:0; }
                else               { dist=tMC.z; tMC.z+=tDC.z; sc.z+=st.z; face=st.z>0?5:4; }
            } else {
                if (tMC.y < tMC.z) { dist=tMC.y; tMC.y+=tDC.y; sc.y+=st.y; face=st.y>0?3:2; }
                else               { dist=tMC.z; tMC.z+=tDC.z; sc.z+=st.z; face=st.z>0?5:4; }
            }
            fineValid = false;
        }
    }

    outDist = dist;
    outFace = face;
    outVox  = mp;
    return hit;
}

[numthreads(8,8,1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= screenW || tid.y >= screenH) return;

    float asp = float(screenW) / float(max(1u, screenH));
    float px  = (2.0*(float(tid.x)+0.5)/float(screenW)-1.0) * asp * tanHalfFov;
    float py  = (1.0-2.0*(float(tid.y)+0.5)/float(screenH)) * tanHalfFov;
    float3 rd = normalize(camFwd + camRight*px + camUp*py);

    float dist; int face; int3 mp;
    bool hit = Trace(camPos, rd, maxDist, dist, face, mp);

    float4 color;
    if (hit) {
        uint bt = GetBlock(mp);
        float3 c = BlockColor(bt, face, mp) * faceBri[face];

        // ---- World-space ambient occlusion (6 fixed secondary rays) ----
        float3 N = faceN[face];
        float3 T = faceT[face];
        float3 B = faceB[face];
        float3 hitPos   = camPos + rd * dist;
        float3 aoOrigin = hitPos + N * AO_BIAS;

        float occ = 0.0;
        [loop] for (int s = 0; s < AO_SAMPLES; ++s)
        {
            float3 h     = AO_DIRS[s];
            float3 aoDir = T * h.x + B * h.y + N * h.z;
            float  aoDist; int aoFace; int3 aoVox;
            if (Trace(aoOrigin, aoDir, AO_MAX_DIST, aoDist, aoFace, aoVox))
                occ += saturate(1.0 - aoDist / AO_MAX_DIST);
        }
        float ao = 1.0 - occ / float(AO_SAMPLES);

        // ---- Directional AO modulation: sun-facing surfaces receive less AO ----
        float NdotL = dot(N, AO_SUN_DIR);
        float sunFacing = saturate(0.5 - 0.5 * NdotL);     // 0 = facing sun, 1 = away
        float dirMod = lerp(1.0, sunFacing, AO_SUN_INFLUENCE);
        float effStrength = AO_STRENGTH * dirMod;
        c *= max(AO_MIN_BRIGHT, (1.0 - effStrength) + effStrength * ao);

        float fogT = saturate(dist/maxDist); fogT *= fogT;
        c = lerp(c, FOG_CLR, fogT);
        color = float4(c, 1.0);
    } else {
        float t = rd.y*0.5+0.5;
        color = float4(lerp(float3(0.67,0.80,0.92), float3(0.25,0.45,0.75), saturate(t)), 1);
    }
    Output[tid.xy] = color;
}
)";