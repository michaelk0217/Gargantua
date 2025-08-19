
layout(binding = 0) RWTexture2D<float4> storageImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    storageImage.GetDimensions(width, height);

    int2 pixelCoords = int2(dispatchThreadID.xy);

    float4 color = float4(
        float(pixelCoords.x) / float(width),
        float(pixelCoords.y) / float(height),
        0.5,
        1.0
    );

    storageImage[pixelCoords] = color;
}