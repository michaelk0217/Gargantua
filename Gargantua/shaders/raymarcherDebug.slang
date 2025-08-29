// raymarcher.slang
struct ShaderData
{
    column_major float4x4 projectionMatrix;
    column_major float4x4 modelMatrix;
    column_major float4x4 viewMatrix;
    column_major float4x4 inverseProjectionMatrix;
    column_major float4x4 inverseViewMatrix;
    float3 cameraPosition;
};

layout(binding = 0) RWTexture2D<float4> storageImage;
layout(binding = 1) ConstantBuffer<ShaderData> ubo;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    storageImage.GetDimensions(width, height);
    int2 pixelCoords = int2(dispatchThreadID.xy);
    if (pixelCoords.x >= width || pixelCoords.y >= height) return;
    
    bool isCenter = (pixelCoords.x == width/2) && (pixelCoords.y == height/2);
    
    float2 uv = (float2(pixelCoords) + 0.5f) / float2(width, height);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y; 
    

    float4 rayNDC = float4(ndc.x, ndc.y, 0.0, 1.0); 
    
    float4 rayView = mul(ubo.inverseProjectionMatrix, rayNDC);
    rayView = rayView / rayView.w;
    
    float4 rayWorld = mul(ubo.inverseViewMatrix, float4(rayView.xyz, 1.0));
    
    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDir = normalize(rayWorld.xyz - rayOrigin);
    
    float3 color = rayDir * 0.5 + 0.5;
    
    if (abs(int(pixelCoords.x) - int(width/2)) <= 1 && pixelCoords.y == height/2)
        color = float3(1.0, 0.0, 0.0); // Horizontal line - red
    else if (pixelCoords.x == width/2 && abs(int(pixelCoords.y) - int(height/2)) <= 1)
        color = float3(1.0, 0.0, 0.0); // Vertical line - red
    else if (isCenter)
        color = rayDir * 0.5 + 0.5; // Center pixel shows actual ray direction
    else
        color = rayDir * 0.5 + 0.5; // All other pixels show their ray direction
    
    storageImage[pixelCoords] = float4(color, 1.0);
}