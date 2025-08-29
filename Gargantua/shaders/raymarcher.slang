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
    int backgroundType; // 0 = Procedural, 1 = HDRI, 2 = color gradient
    float exposure;
    float gamma;

    // black hole params for ui
    float blackHoleMass;
    float blackHoleSpin;
    int maxSteps;
    float stepSize;
    int geodesicType;

    // accretion disk params
    int diskEnable;
    int volSubsteps;
    float dsVolScale;
    float sigmaT;
    
    float diskRin; // inner radius;
    float diskRout; // outer radius
    float diskH; // vertical scale height
    float diskEdgeK; // softness of inner/outer edge

    float diskDensity;
    float diskNoiseAmp;
    float diskNoiseScale;
    float diskNoiseWarp;
    
    float emissionScale; // overall disk brightness
    float vScale; // Keplerian seed multiplier
    float dopplerPower;
    float temperatureBias;

};

layout(binding = 0) RWTexture2D<float4> hdrColorBuffer;
layout(binding = 1) ConstantBuffer<ShaderData> ubo;
layout(binding = 2) Texture2D<float4> spheremapTexture;
layout(binding = 3) SamplerState environmentSampler;
layout(binding = 4) Texture3D<float4> noiseTexture;
layout(binding = 5) SamplerState noiseSampler;


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
        sampleProcedural(direction, color);

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


void sampleProcedural(float3 direction, out float3 color)
{
    // Procedural starfield
    float2 sphericalUV = directionToSphericalUV(direction);

    float stars = 0.0;
    for (int i = 0; i < 10; i++) {
        float2 gridUV = sphericalUV * (50.0 + i * 30.0);
        float2 gridID = floor(gridUV);
        float2 gridPos = frac(gridUV) - 0.5;

        float rand = frac(sin(dot(gridID, float2(12.9898, 78.233))) * 43758.5453);

        if (rand > 0.98) {
            float dist = length(gridPos);
            float star = 1.0 - smoothstep(0.0, 0.05 / (1.0 + i), dist);
            stars += star * (0.5 + 0.5 * rand);
        }
    }

    float milkyway = 1.0 - abs(direction.y);
    milkyway = pow(milkyway, 4.0) * 0.3;

    float3 deepSpace = lerp(
        float3(0.0, 0.0, 0.03),
        float3(0.0, 0.0, 0.1),
        abs(direction.y)
    );

    // color = deepSpace + float3(stars) + float3(0.4, 0.3, 0.6) * milkyway;
    // color = deepSpace + float3(stars) + float3(0.4, 0.3, 0.6);
    color = deepSpace + float3(stars);
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

// ------ Kerr Metric Physics Helpers ------
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

void sampleDisk(
    float3 p,
    float3 rayDir,
    float M,
    float a,
    float time,
    Texture3D<float4> noiseTexture,
    SamplerState noiseSampler,
    out float density,
    out float3 emissive
)
{
    // cylindrical coords
    float2 xz = float2(p.x, p.z);
    float r = length(xz);
    float h = p.y;
    float theta = atan2(xz.y, xz.x);

    // ----- density calc -----
    float innerRadius = ubo.diskRin * M;
    float outerRadius = ubo.diskRout * M;
    float innerFade = smoothstep(innerRadius, innerRadius + M, r);
    float outerFade = 1.0 - smoothstep(outerRadius - ubo.diskEdgeK, outerRadius, r);
    float radialMask = innerFade * outerFade;

    float diskHeight = ubo.diskH * (1.0 + 0.3 * sqrt(r / (10.0 * M))); 
    float verticalProfile = exp(-(h * h) / (2.0 * diskHeight * diskHeight));

    float orbitalVelocity = sqrt(M / max(r, 0.001));
    // float orbitalPeriod = 2.0 * 3.14159 * r / max(orbitalVelocity, 0.001);
    // float rotationSpeed = 1.0 / pow(r, 1.5);

    float rotationRate = orbitalVelocity / r;
    float windingNumber = 2.0;

    float spiralAngle = theta - rotationRate * time * 10.0;
    // float spiralAngle = theta - rotationRate * time * ubo.blackHoleSpin;

    spiralAngle += windingNumber * log(r / innerRadius);

    float3 spiralCoords;
    spiralCoords.x = cos(spiralAngle) * (r - innerRadius) / (outerRadius - innerRadius);
    spiralCoords.y = h / diskHeight;
    spiralCoords.z = sin(spiralAngle) * (r - innerRadius) / (outerRadius - innerRadius);

    float noise1 = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 1.0, 0).r;
    float noise2 = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 3.0, 0).r;
    float noise3 = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 7.0, 0).r;

    float spiralPattern = sin(spiralAngle * windingNumber) * 0.5 + 0.5;
    spiralPattern = pow(spiralPattern, 2.0);



    float turbulence = noise1 + 0.3 * noise2 + 0.1 * noise3;
    turbulence = turbulence / 1.4;

    
    float structure = lerp(turbulence, spiralPattern * turbulence, 0.5);
    
    float radialDensityBoost = 1.0 + 2.0 * exp(-(r - innerRadius) / (3.0 * M));
    
    density = ubo.diskDensity * radialMask * verticalProfile * structure * radialDensityBoost;
    
    float clumpiness = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 0.5 + time * 0.1, 0).r;
    if (clumpiness > 0.7) {
        density *= 1.0 + (clumpiness - 0.7) * 3.0;
    }

    // ----- color/emission calculation -----

    float temperatureParam = 1.0 - saturate((r - innerRadius) / (outerRadius - innerRadius));
    temperatureParam = pow(temperatureParam, 2.0);

    // ring color todo: make a UI param
    float3 coolColor = float3(0.1, 0.3, 0.9); 
    float3 warmColor = float3(0.1, 0.3, 0.9);
    float3 hotColor = float3(0.4, 0.6, 1.0);
    float3 veryHotColor = float3(0.7, 0.8, 1.0); 

    float3 diskColor;
    if (temperatureParam < 0.25) {
        diskColor = lerp(coolColor, warmColor, temperatureParam * 4.0);
    } else if (temperatureParam < 0.5) {
        diskColor = lerp(warmColor, hotColor, (temperatureParam - 0.25) * 4.0);
    } else if (temperatureParam < 0.75) {
        diskColor = lerp(hotColor, veryHotColor, (temperatureParam - 0.5) * 4.0);
    } else {
        diskColor = lerp(veryHotColor, float3(0.7, 0.8, 1.0), (temperatureParam - 0.75) * 4.0);
    }

    // ----- doppler effects -----
    float3 tangent = normalize(float3(-xz.y, 0, xz.x));
    float3 velocity = tangent * orbitalVelocity;
    
    float beta = length(velocity);
    float gamma = 1.0 / sqrt(max(1.0 - beta * beta, 0.001));

    float cosAngle = dot(normalize(velocity), -rayDir);
    float dopplerFactor = gamma * (1.0 - beta * cosAngle);
    // relativistic beaming
    float dopplerBoost = pow(1.0 / max(dopplerFactor, 0.1), ubo.dopplerPower);  

    if (cosAngle > 0) {
        // moving toward observer - blueshift
        diskColor = lerp(diskColor, diskColor * float3(0.7, 0.85, 1.2), cosAngle * 0.3);
        emissive = diskColor * (1.0 + cosAngle * 0.5); // Brighter
    } else {
        // moving away - redshift
        diskColor = lerp(diskColor, diskColor * float3(1.2, 0.85, 0.7), -cosAngle * 0.3);
        emissive = diskColor * (1.0 + cosAngle * 0.3); // Dimmer
    }

    // ----- gravitational effects -----

    float gravitationalRedshift = sqrt(1.0 - 2.0 * M / max(r, 2.0 * M));
    float compressionBoost = 1.0 + 1.0 * exp(-r / (5.0 * M));
    
    emissive *= ubo.emissionScale * density * dopplerBoost * compressionBoost * gravitationalRedshift;
    
    // Hot spots and flares
    float flareNoise = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 0.3 + time * 0.5, 0).r;
    if (flareNoise > 0.85 && temperatureParam > 0.6) {
        emissive *= 1.0 + (flareNoise - 0.85) * 5.0;
    }
}

// float computeTransmittance(
//     float3 position,
//     float3 lightDir,
//     float stepSize,
//     int steps,
//     float M,
//     float a,
//     float time,
//     Texture3D<float4> noiseTexture,
//     SamplerState noiseSampler
// )
// {
//     float tau = 0.0;
    
//     for (int i = 0; i < steps; i++) {
//         float3 samplePos = position + lightDir * (float(i) * stepSize);
        
//         float density;
//         float3 emissive;  // Not used here
//         sampleDisk(samplePos, -lightDir, M, a, time, noiseTexture, noiseSampler, density, emissive);
        
//         tau += density * stepSize;
//     }
    
//     return exp(-tau);
// }

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width, height;
    hdrColorBuffer.GetDimensions(width, height);
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

    // geodesic / BH setup
    const float M = ubo.blackHoleMass;
    const float a = ubo.blackHoleSpin * M; // spin parameter
    int MAX_STEPS = ubo.maxSteps;
    float STEP_SIZE = ubo.stepSize;
    const float MAX_DISTANCE = 200.0;
    const float horizonRadius = M + sqrt(M * M - a * a);

    // ray state
    RayState rayState;
    rayState.position = rayOrigin;
    rayState.direction = rayDir;
    rayState.redshift = 1.0;

    // volumetric accumulators
    float3 L = float3(0, 0, 0); // emitted radiance along the path
    float T = 1.0; // transmittance

    float2 noiseCoord = float2(pixelCoords) / 64.0;
    float blueNoise = frac(sin(dot(noiseCoord, float2(12.9898, 78.233))) * 43758.5453);
    float jitter = (blueNoise - 0.5) * STEP_SIZE * 0.5;

    bool captured = false;

    rayState.position += rayState.direction * jitter;


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
        rayState.redshift *= 1.0 / max(metric.alpha, 1e-3);
        // maybe todo: use camera's 4-velocity relative to FIDO and calculate doppler redshifts

        // predict next geodesic state
        RayState nextState = rayState;
        if (ubo.geodesicType == 0)
        {
            nextState = integrateGeodesicSimple(nextState, STEP_SIZE, a, M);
        }
        else
        {
            nextState = integrateGeodesicRK4(nextState, STEP_SIZE, a, M);
        }


        if (ubo.diskEnable != 0 && T > 0.01)
        {
            float3 seg = nextState.position - rayState.position;
            float segLen = length(seg);

            if (segLen > 0.0)
            {
                float3 segDir = seg / segLen;
                int sub = max(1, ubo.volSubsteps);
                float ds = (segLen * ubo.dsVolScale) / sub;

                // midpoint sampling per micro-step
                for (int s = 0; s < sub; ++s)
                {
                    float tMid = (float(s) + 0.5) / sub;
                    float3 pMid = rayState.position + seg * tMid;
                    
                    float2 xz = float2(pMid.x, pMid.z);
                    float rperp = length(xz);
                    float h = abs(pMid.y);

                    if (rperp >= ubo.diskRin * 0.8 && rperp <= ubo.diskRout * 1.2 && h <= ubo.diskH * 5.0)
                    {
                        float density;
                        float3 emissive;
                        sampleDisk(
                            pMid,
                            segDir,
                            M,
                            a,
                            ubo.time,
                            noiseTexture,
                            noiseSampler,
                            density,
                            emissive
                        );

                        if (density > 0.001)
                        {
                            float extinction = ubo.sigmaT * density * ds;
                            float atten = exp(-extinction);

                            float3 src = emissive * density;

                            if (extinction > 1e-5)
                            {
                                float w = (1.0 - atten) / extinction;
                                L += T * src * ubo.sigmaT * w * ds;
                            }
                            else
                            {
                                L += T * src * ubo.sigmaT * ds;
                            }

                            T *= atten;
                            if (T < 0.01) break;
                        }
                    }
                }

                if (T < 0.01) break;
            }
        }

        rayState = nextState;
    }

    float3 finalColor;
    if (captured)
    {
        if (ubo.diskEnable == 0)
        {
            finalColor = float3(0.0, 0.0, 0.0);
        }
        else
        {
            // float3 rimGlow = float3(0.0, 0.0, 0.0);
            // float lastDist = length(rayState.position);
            // if (lastDist < horizonRadius * 1.5)
            // {
            //     float glowStrength = 1.0 - saturate((lastDist - horizonRadius) / (horizonRadius * 0.5));
            //     rimGlow = float3(0.5, 0.3, 0.8) * glowStrength * 0.2; // Purple-ish glow
            // }
            finalColor = L;
        }
    }
    else
    {
        float3 bg = sampleBackground(rayState.direction);
        // if (ubo.backgroundType == 1)
        // {
        //     float exposure = ubo.exposure; // default 1.0
        //     float gamma = ubo.gamma; // default 2.2
            
        //     // bg = reinhardToneMapping(bg, exposure);
        //     // bg = exposureToneMapping(bg, exposure);
        //     bg = ACESFilm(bg * exposure);
        //     bg = pow(bg, float3(1.0 / gamma));
        // }
        
        finalColor = L + T * bg;
    }

    hdrColorBuffer[pixelCoords] = float4(finalColor, 1.0);
}