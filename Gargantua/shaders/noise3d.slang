// noise3d.slang

struct NoiseParams
{
    int size;
    float scale;
    float boost;
    int octaves;
    int tileSize;
    float lacunarity;
    float gain;
    float time;
};

layout(binding = 0) RWTexture3D<float4> outNoise;
layout(binding = 1) ConstantBuffer<NoiseParams> params;

uint hash3D(int3 p)
{
    uint h = uint(p.x * 374761393u + p.y * 668265263u + p.z * 3624371u);
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

float3 perlinGradient(int3 p, int period)
{
    int3 ip = (p % period + period) % period;                 
    float rand = frac(float(hash3D(ip)) * 0.00000011920928955078125); // 1/2^23
    float theta = rand * 6.2831853;
    float phi = acos(2.0 * frac(rand * 1.37) - 1.0);
    return float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

float3 fade(float3 t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlinNoise3D(float3 p, int period)
{
    float3 pi = floor(p);
    float3 pf = frac(p);
    
    float dots[8];
    for (int i = 0; i < 8; i++)
    {
        int3 corner = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        float3 grad = perlinGradient(int3(pi) + corner, period);
        float3 dist = pf - float3(corner);
        dots[i] = dot(grad, dist);
    }
    
    float3 f = fade(pf);
    
    // trilinear interpolation
    float x1 = lerp(dots[0], dots[1], f.x);
    float x2 = lerp(dots[2], dots[3], f.x);
    float x3 = lerp(dots[4], dots[5], f.x);
    float x4 = lerp(dots[6], dots[7], f.x);
    
    float y1 = lerp(x1, x2, f.y);
    float y2 = lerp(x3, x4, f.y);
    
    return lerp(y1, y2, f.z);
}

float fbm3(float3 p, int octaves, float lacunarity, float gain, int basePeriod)
{
    float value = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float totalAmp = 0.0;
    
    for (int i = 0; i < octaves; i++)
    {
        int period = int(float(basePeriod) * freq);
        value += perlinNoise3D(p * freq, period) * amp;
        totalAmp += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    
    return value / totalAmp;  // normalize
}

[numthreads(8, 8, 8)]
[shader("compute")]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int size = params.size;
    if (dispatchThreadID.x >= size || dispatchThreadID.y >= size || dispatchThreadID.z >= size) return;

    float3 uv = (float3(dispatchThreadID) + 0.5) / float(size);
    uv *= params.scale;

    float value = fbm3(uv, params.octaves, params.lacunarity, params.gain, params.tileSize);
    value = saturate(value + 0.5);

    value *= params.boost;
    value = max(value, 0.0);

    outNoise[dispatchThreadID] = float4(value, value, value, 1.0);

}