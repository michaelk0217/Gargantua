// post_composite.slang

// struct CompositeParams {
//     float threshold;     // Unused in composite
//     float softKnee;      // Unused in composite
//     float bloomIntensity;
//     int mipLevel;        // Unused in composite
    
//     int blurDirection;   // Unused in composite
//     float texelSizeX;    // Unused in composite
//     float texelSizeY;    // Unused in composite
//     float padding1;
    
//     float exposure;
//     float gamma;
//     float padding[2];
// };

layout(binding = 0) Texture2D<float4> hdrTexture;
layout(binding = 1) Texture2D<float4> bloomTextures[5];  // Array of mips
layout(binding = 2) RWTexture2D<float4> outputTexture;
layout(binding = 3) SamplerState linearSampler;
// layout(binding = 4) ConstantBuffer<CompositeParams> params;

[push_constant]
cbuffer PerDrawConstants
{
    float exposure;
    float gamma;
    float padding[2];
};

float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    
    float3 hdr = hdrTexture.SampleLevel(linearSampler, uv, 0).rgb;
    
    // Accumulate bloom from mip chain
    float3 bloom = float3(0, 0, 0);
    for (int i = 0; i < 5; i++) {
        float3 mipBloom = bloomTextures[i].SampleLevel(linearSampler, uv, i).rgb;
        bloom += mipBloom * (1.0 / (i + 1)); // Weight by mip level
    }
    // float3 combined = hdr + bloom * params.bloomIntensity;
    float3 combined = hdr + bloom;
    
    float3 mapped = ACESFilm(combined * exposure);
    float3 final = pow(mapped, float3(1.0 / gamma));
    
    outputTexture[id.xy] = float4(final, 1.0);
}