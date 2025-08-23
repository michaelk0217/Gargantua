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

    // black hole params for ui
    float blackHoleMass;
    float blackHoleSpin;
    int maxSteps;
    float stepSize;
    int geodesicType;
};

layout(binding = 0) RWTexture2D<float4> storageImage;
layout(binding = 1) ConstantBuffer<ShaderData> ubo;
layout(binding = 2) Texture2D<float4> spheremapTexture;
layout(binding = 3) SamplerState environmentSampler;


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
            // float3(0.02, 0.02, 0.05),
            // float3(0.05, 0.03, 0.1),
            // float3(0.22, 0.13, 0.30),
            // float3(0.09, 0.09, 0.26),
            float3(0.0, 0.0, 0.02),      // Near-black with a blue tint (for the horizon)
            float3(0.01, 0.02, 0.08),     // Very dark, desaturated navy blue (for poles)
            abs(direction.y)
        );
        
        color = deepSpace + float3(stars) + float3(0.4, 0.3, 0.6) * milkyway;
        color = color.bgr;
    }
    else if (ubo.backgroundType == 1) // spheremap hdr
    {
        float2 uv = directionToEquirectangularUV(direction);
        
        color = spheremapTexture.SampleLevel(environmentSampler, uv, 0).rgb;
    }
    else if (ubo.backgroundType == 2)
    {
         // //rainbow
        color = direction * 0.5 + 0.5;
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

// ------ Kerr Metric Physics ------
struct KerrMetricComponents
{
    float alpha;    // lapse function
    float omega;    // frame dragging
    float varpi;    // cylindrical radius
    float rho2;
    float Delta;
    float Sigma;
};

KerrMetricComponents computeKerrMetric(float r, float theta, float a, float M)
{
    KerrMetricComponents comp;

    // DNGR paper A.1
    comp.rho2 = r * r + a * a * cos(theta) * cos(theta);
    comp.Delta = r * r - 2.0 * M * r + a * a;
    comp.Sigma = sqrt(pow(r * r + a * a, 2.0) - a * a * comp.Delta * sin(theta) * sin(theta));
    
    comp.alpha = sqrt(comp.rho2 * comp.Delta) / comp.Sigma;
    comp.omega = 2.0 * a * r / (comp.Sigma * comp.Sigma);
    comp.varpi = comp.Sigma * sin(theta) / sqrt(comp.rho2);

    return comp;
}

float2 cartesianToSpherical(float3 pos)
{
    float r = length(pos);
    float theta = acos(pos.y / r);
    float phi = atan2(pos.z, pos.x);
    return float2(theta, phi);
}

// struct GeodesicState
// {
//     float3 position;
//     float3 direction;
//     float energy;
//     float angularMomentum;
//     float carterConstant;
// };

struct RayState
{
    float3 position;
    float3 direction;
    float redshift;
};

// simplified for real time (omitting energy, angular momentum, and carter constant)
float3 computeGravitationalAcceleration(float3 position, float a, float M)
{
    float r = length(position);
    if (r < 1e-3) return float3(0, 0, 0);

    float2 spherical = cartesianToSpherical(position);
    float theta = spherical.x;

    KerrMetricComponents metric = computeKerrMetric(r, theta, a, M);

    float3 radialDir = normalize(position);
    float gravitationalAccel = -1.5 * M / (r * r);

    // frame dragging
    float3 spinAxis = float3(0.0, 1.0, 0.0);
    float3 tangentialDir = normalize(cross(spinAxis, radialDir));
    float frameDragAccel = a * M / (r * r * r);

    return gravitationalAccel * radialDir + frameDragAccel * tangentialDir;
}

// Proper geodesic integration using 4th order Runge-Kutta
RayState integrateGeodesicRK4(RayState state, float stepSize, float a, float M)
{
    float3 pos = state.position;
    float3 dir = state.direction;

    // k1 = f(t, y)
    float3 k1_pos = dir;
    float3 k1_dir = computeGravitationalAcceleration(pos, a, M);

    // k2 = f(t + h/2, y + h*k1/2)
    float3 k2_pos = dir + 0.5 * stepSize * k1_dir;
    float3 k2_dir = computeGravitationalAcceleration(pos + 0.5 * stepSize * k1_pos, a, M);

    // k3 = f(t + h/2, y + h*k2/2)
    float3 k3_pos = dir + 0.5 * stepSize * k2_dir;
    float3 k3_dir = computeGravitationalAcceleration(pos + 0.5 * stepSize * k2_pos, a, M);

    // k4 = f(t + h, y + h*k3)
    float3 k4_pos = dir + stepSize * k3_dir;
    float3 k4_dir = computeGravitationalAcceleration(pos + stepSize * k3_pos, a, M);

    state.position += (stepSize / 6.0) * (k1_pos + 2.0*k2_pos + 2.0*k3_pos + k4_pos);
    state.direction += (stepSize / 6.0) * (k1_dir + 2.0*k2_dir + 2.0*k3_dir + k4_dir);
    state.direction = normalize(state.direction);
    
    return state;
}

// Simplified version for better real-time performance
RayState integrateGeodesicSimple(RayState state, float stepSize, float a, float M)
{
    float3 acceleration = computeGravitationalAcceleration(state.position, a, M);
    
    // Update direction first (Euler integration)
    state.direction += acceleration * stepSize;
    state.direction = normalize(state.direction);
    
    // Then update position
    state.position += state.direction * stepSize;
    
    return state;
}

float computeRedshift(float r, float theta, float a, float M, float3 cameraPos, float3 rayDir)
{
    KerrMetricComponents metric = computeKerrMetric(r, theta, a, M);
    
    // Gravitational redshift: 1/α
    float gravitationalRedshift = 1.0 / metric.alpha;
    
    // simplified - should be relative to cmaera 4-vel
    float dopplerEffect = 1.0 + 0.1 * dot(normalize(cameraPos), rayDir);
    
    return gravitationalRedshift * dopplerEffect;
}

float3 applyFrequencyShifts(float3 color, float redshift)
{
    if (redshift > 1.0)
    {
        // redshift
        color = lerp(color, float3(color.r * 1.8, color.g * 0.6, color.b * 0.4), saturate(1.0 - redshift));
    }
    else
    {
        color = lerp(color, float3(color.r * 0.4, color.g * 0.6, color.b * 1.8), saturate(1.0 - redshift));
    }
    return color;
}


[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    storageImage.GetDimensions(width, height);
    int2 pixelCoords = int2(dispatchThreadID.xy);

    if (pixelCoords.x >= width || pixelCoords.y >= height) return;
    
    float2 uv = (float2(pixelCoords) + 0.5f) / float2(width, height);
    uv = uv * 2.0f - 1.0f;
    uv.y = -uv.y; // accounting for vulkan's screen coords are top-to-bottom

    float4 worldPos = float4(uv, 1.0, 1.0);
    worldPos = mul(ubo.inverseProjectionMatrix, worldPos);
    worldPos /= worldPos.w;
    worldPos = mul(ubo.inverseViewMatrix, float4(worldPos.xyz, 0.0));

    float3 rayOrigin = ubo.cameraPosition;
    float3 rayDir = normalize(worldPos.xyz);

    const float M = ubo.blackHoleMass;
    const float a = ubo.blackHoleSpin * M; // spin parameter
    int MAX_STEPS = ubo.maxSteps;
    float STEP_SIZE = ubo.stepSize;
    const float MAX_DISTANCE = 200.0;

    const float horizonRadius = M + sqrt(M * M - a * a);

    RayState rayState;
    rayState.position = rayOrigin;
    rayState.direction = rayDir;
    rayState.redshift = 1.0;

    bool captured = false;

    for (int i = 0; i < MAX_STEPS; i++)
    {
        float distToOrigin = length(rayState.position);

        if (distToOrigin < horizonRadius)
        {
            captured = true;
            break;
        }

        if (length(rayState.position - rayOrigin) > MAX_DISTANCE) {
            break;
        }

        float2 spherical = cartesianToSpherical(rayState.position);
        KerrMetricComponents metric = computeKerrMetric(distToOrigin, spherical.x, a, M);
        rayState.redshift *= 1.0 / metric.alpha;
        // maybe todo: use camera's 4-velocity relative to FIDO and calculate doppler redshifts

        if (ubo.geodesicType == 0)
        {
            rayState = integrateGeodesicSimple(rayState, STEP_SIZE, a, M);
        }
        else
        {
            rayState = integrateGeodesicRK4(rayState, STEP_SIZE, a, M);
        }

    }

    float3 finalColor;
    if (captured)
    {
        finalColor = float3(0.0, 0.0, 0.0);
    }
    else
    {
        finalColor = sampleBackground(rayState.direction);
        if (ubo.backgroundType == 1)
        {
            float exposure = ubo.exposure; // default 1.0
            float gamma = ubo.gamma; // default 2.2
            
            // finalColor = reinhardToneMapping(finalColor, exposure);
            // finalColor = exposureToneMapping(finalColor, exposure);
            finalColor = ACESFilm(finalColor * exposure);

            finalColor = pow(finalColor, float3(1.0 / gamma));
        }
    }

    storageImage[pixelCoords] = float4(finalColor, 1.0);
}