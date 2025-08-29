// gaussian_blur.slang

// struct BlurParams {
//     float threshold;     // Unused in this shader
//     float softKnee;      // Unused in this shader
//     float intensity;     // Unused in this shader
//     int mipLevel;
    
//     int blurDirection;   // 0 = vertical, 1 = horizontal
//     float texelSizeX;
//     float texelSizeY;
//     float padding1;
    
//     float exposure;      // Unused in this shader
//     float gamma;         // Unused in this shader
//     float padding2[2];
// };



layout(binding = 0) Texture2D<float4> inputTexture;
layout(binding = 1) RWTexture2D<float4> outputTexture;
layout(binding = 2) SamplerState linearSampler;
// layout(binding = 3) ConstantBuffer<BlurParams> params;

[push_constant]
cbuffer PerDrawConstants
{
    int mipLevel;
    int blurDirection;   // 0 = vertical, 1 = horizontal
    float texelSizeX;
    float texelSizeY;
};

// gaussian weights
static const float weights[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    
    float2 texelSize = blurDirection == 1 ? float2(texelSizeX, 0.0) : float2(0.0, texelSizeY);
    
    float4 result = inputTexture.SampleLevel(linearSampler, uv, 0) * weights[0];
    
    [unroll]
    for (int i = 1; i < 5; i++) {
        float2 offset = texelSize * float(i);
        result += inputTexture.SampleLevel(linearSampler, uv + offset, 0) * weights[i];
        result += inputTexture.SampleLevel(linearSampler, uv - offset, 0) * weights[i];
    }
    
    outputTexture[id.xy] = result;
}