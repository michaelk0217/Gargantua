// bloom_extract.slang
// struct Params {
//     float threshold;
//     float softKnee;
//     float intensity;
//     int mipLevel; // Unused in this shader
    
//     int blurDirection;  // Unused in this shader
//     float texelSizeX;   // Unused in this shader
//     float texelSizeY;   // Unused in this shader
//     float padding1;
    
//     float exposure;     // Unused in this shader
//     float gamma;        // Unused in this shader
//     float padding2[2];
// };

// struct BloomParams {
//     float threshold;
//     float softKnee;
//     float intensity;
//     float padding;
// };

layout(binding = 0) Texture2D<float4> inputTexture;
layout(binding = 1) RWTexture2D<float4> outputTexture;
layout(binding = 2) SamplerState linearSampler;
// layout(binding = 3) ConstantBuffer<Params> params;

[push_constant]
cbuffer PerDrawConstants
{
    float threshold;
    float softKnee;
    float intensity;
    float padding;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height) return;
    
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float4 color = inputTexture.SampleLevel(linearSampler, uv, 0);
    
    float luminance = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
    
    // Soft threshold
    // float knee = params.threshold * params.softKnee;
    // float soft = luminance - params.threshold + knee;
    // soft = clamp(soft, 0, 2 * knee);
    // soft = soft * soft / (4 * knee + 0.00001);
    // float contribution = max(soft, luminance - params.threshold) / max(luminance, 0.00001);
    
    // outputTexture[id.xy] = color * contribution * params.intensity;

    float knee = threshold * softKnee;
    float soft = luminance - threshold + knee;
    soft = clamp(soft, 0, 2 * knee);
    soft = soft * soft / (4 * knee + 0.00001);
    float contribution = max(soft, luminance - threshold) / max(luminance, 0.00001);

    outputTexture[id.xy] = color * contribution * intensity;
}
