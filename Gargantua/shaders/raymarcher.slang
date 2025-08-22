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

// float hit_sphere(float3 center, float radius, float3 rayOrigin, float3 rayDirection) {
//     float3 oc = center - rayOrigin;
//     float a = dot(rayDirection, rayDirection);
//     float b = -2.0 * dot(rayDirection, oc);
//     float c = dot(oc, oc);
//     float discriminant = b * b - 4 * a * c;

//     if (discriminant < 0) {
//         return -1.0;
//     } else {
//         return (-b - sqrt(discriminant)) / (2.0 * a);
//     }
// }


[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    storageImage.GetDimensions(width, height);
    int2 pixelCoords = int2(dispatchThreadID.xy);

    if (pixelCoords.x >= width || pixelCoords.y >= height) return;
    
    // pixel coords -> normalized device coords
    float2 uv = (float2(pixelCoords) + 0.5f) / float2(width, height);
    uv = uv * 2.0f - 1.0f;
    uv.y = -uv.y; // accounting for vulkan's screen coords are top-to-bottom



    float4 worldPos = float4(uv, 1.0, 1.0);
    worldPos = mul(ubo.inverseProjectionMatrix, worldPos);
    worldPos /= worldPos.w;

    worldPos = mul(ubo.inverseViewMatrix, float4(worldPos.xyz, 0.0));

    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDir = normalize(worldPos.xyz);


    // ray marching
    const int MAX_STEPS = 200;
    const float STEP_SIZE = 0.1;
    // const float HIT_DISTANCE = 0.01;
    const float MAX_DISTANCE = 100.0;
    const float SCHWARZSCHILD_RADIUS = 1.0f; // event horizon

    float3 p = rayOrigin;
    bool captured = false;

    for (int i = 0; i < MAX_STEPS; i++)
    {
        float distToOrigin = length(p);

        if (distToOrigin < SCHWARZSCHILD_RADIUS)
        {
            captured = true;
            break;
        }

        float3 acceleration = -1.5 * SCHWARZSCHILD_RADIUS * p / (distToOrigin * distToOrigin * distToOrigin);
        rayDir += acceleration * STEP_SIZE;
        rayDir = normalize(rayDir);

        p += rayDir * STEP_SIZE;

        if (length(p - rayOrigin) > MAX_DISTANCE) {
            break;
        }
    }

    float3 finalColor;
    if (captured)
    {
        finalColor = float3(0.0, 0.0, 0.0);
    }
    else
    {
        finalColor = rayDir * 0.5 + 0.5;
    }

    storageImage[pixelCoords] = float4(finalColor, 1.0);
}