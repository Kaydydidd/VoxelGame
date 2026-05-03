#include "TerrainNoise.h"

#include <cmath>
#include <cstdint>

namespace {

    float Lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    float Smooth(float t) {
        return t * t * (3.0f - 2.0f * t);
    }

    uint32_t Hash2D(int x, int y) {
        uint32_t h = 2166136261u;
        h = (h ^ uint32_t(x)) * 16777619u;
        h = (h ^ uint32_t(y)) * 16777619u;
        h ^= h >> 13;
        h *= 1274126177u;
        h ^= h >> 16;
        return h;
    }

    float Random01(int x, int y) {
        return float(Hash2D(x, y) & 0x00FFFFFF) / float(0x00FFFFFF);
    }

    float ValueNoise(float x, float y) {
        int x0 = int(std::floor(x));
        int y0 = int(std::floor(y));

        float sx = Smooth(x - float(x0));
        float sy = Smooth(y - float(y0));

        float a = Lerp(Random01(x0, y0), Random01(x0 + 1, y0), sx);
        float b = Lerp(Random01(x0, y0 + 1), Random01(x0 + 1, y0 + 1), sx);

        return Lerp(a, b, sy);
    }

} // namespace

float FBM(float x, float y, int oct) {
    float s = 0.0f;
    float a = 0.5f;
    float f = 1.0f;
    float n = 0.0f;

    for (int i = 0; i < oct; ++i) {
        s += ValueNoise(x * f, y * f) * a;
        n += a;
        a *= 0.5f;
        f *= 2.0f;
    }

    return s / n;
}