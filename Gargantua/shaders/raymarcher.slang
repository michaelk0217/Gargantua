// raymarcher.slang

struct ShaderData
{
    column_major float4x4 projectionMatrix;
    column_major float4x4 modelMatrix;
    column_major float4x4 viewMatrix;
    column_major float4x4 inverseProjectionMatrix;
    column_major float4x4 inverseViewMatrix;
    float3 cameraPosition;
    float time;
    int backgroundType;
    float exposure;
    float gamma;
};

layout(binding = 0) RWTexture2D<float4> storageImage;
layout(binding = 1) ConstantBuffer<ShaderData> ubo;
layout(binding = 2) Texture2D<float4> spheremapTexture;
layout(binding = 3) SamplerState environmentSampler;

// struct GeodesicState
// {
//     float3 position;
//     float3 direction; // 4-momentum normalized
// }

float2 directionToSphericalUV(float3 direction)
{
    const float PI = 3.14159265359;

    float3 d = normalize(direction);
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float phi = atan2(d.z, d.x);

    float u = phi / (2.0 * PI) + 0.5;
    float v = theta / PI;

    return float2(u, v);
}

float2 directionToEquirectangularUV(float3 direction)
{
    const float PI = 3.14159265359;

    float3 d = normalize(direction);
    float theta = asin(clamp(d.y, -1.0, 1.0));
    float phi = atan2(d.z, d.x);

    float u = phi / (2.0 * PI) + 0.5;
    float v = theta / PI + 0.5;
    
    return float2(u, v);
}

float3 sampleBackground(float3 direction)
{
    float3 color = float3(0.0);
    if (ubo.backgroundType == 0) // procedural
    {
        // //rainbow
        // color = direction * 0.5 + 0.5;

        // Procedural starfield
        float2 sphericalUV = directionToSphericalUV(direction);
        
        float stars = 0.0;
        for (int i = 0; i < 5; i++) {
            float2 gridUV = sphericalUV * (50.0 + i * 30.0);
            float2 gridID = floor(gridUV);
            float2 gridPos = frac(gridUV) - 0.5;
            
            // Pseudo-random per grid cell
            float rand = frac(sin(dot(gridID, float2(12.9898, 78.233))) * 43758.5453);
            
            if (rand > 0.98) {
                float dist = length(gridPos);
                float star = 1.0 - smoothstep(0.0, 0.05 / (1.0 + i), dist);
                stars += star * (0.5 + 0.5 * rand);
            }
        }
        
        float milkyway = 1.0 - abs(direction.y);
        milkyway = pow(milkyway, 4.0) * 0.3;
        
        // Deep space blue-purple gradient
        float3 deepSpace = lerp(
            float3(0.02, 0.02, 0.05),
            float3(0.05, 0.03, 0.1),
            abs(direction.y)
        );
        
        color = deepSpace + float3(stars) + float3(0.4, 0.3, 0.6) * milkyway;
    }
    else if (ubo.backgroundType == 1) // spheremap hdr
    {
        float2 uv = directionToEquirectangularUV(direction);
        
        color = spheremapTexture.SampleLevel(environmentSampler, uv, 0).rgb;
    }

    return color;
}

// ------ tone mapping functions for hdr ------
float3 ACESFilm(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 reinhardToneMapping(float3 color, float exposure)
{
    color *= exposure;
    return color / (1.0 + color);
}

float3 exposureToneMapping(float3 color, float exposure)
{
    return float3(1.0) - exp(-color * exposure);
}
// ----------------------------------------------

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
    float3 currentRayDir = rayDir;

    for (int i = 0; i < MAX_STEPS; i++)
    {
        float distToOrigin = length(p);

        if (distToOrigin < SCHWARZSCHILD_RADIUS)
        {
            captured = true;
            break;
        }

        float3 acceleration = -1.5 * SCHWARZSCHILD_RADIUS * p / (distToOrigin * distToOrigin * distToOrigin);
        currentRayDir += acceleration * STEP_SIZE;
        currentRayDir = normalize(currentRayDir);

        p += currentRayDir * STEP_SIZE;

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
        finalColor = sampleBackground(currentRayDir);
        if (ubo.backgroundType == 1)
        {
            float exposure = ubo.exposure; // default 1.0
            float gamma = ubo.gamma; // default 2.2
            
            finalColor = reinhardToneMapping(finalColor, exposure);
            // finalColor = exposureToneMapping(finalColor, exposure);
            // finalColor = ACESFilm(finalColor * exposure);

            finalColor = pow(finalColor, float3(1.0 / gamma));
        }
    }

    storageImage[pixelCoords] = float4(finalColor, 1.0);
}