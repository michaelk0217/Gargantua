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

layout(binding = 0) RWTexture2D<float4> storageImage;
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
            float3(0.0, 0.0, 0.02),
            float3(0.01, 0.02, 0.08),
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

// float computeRedshift(float r, float theta, float a, float M, float3 cameraPos, float3 rayDir)
// {
//     KerrMetricComponents metric = computeKerrMetric(r, theta, a, M);
    
//     // Gravitational redshift: 1/α
//     float gravitationalRedshift = 1.0 / metric.alpha;
    
//     // simplified - should be relative to cmaera 4-vel
//     float dopplerEffect = 1.0 + 0.1 * dot(normalize(cameraPos), rayDir);
    
//     return gravitationalRedshift * dopplerEffect;
// }

// float3 applyFrequencyShifts(float3 color, float redshift)
// {
//     if (redshift > 1.0)
//     {
//         // redshift
//         color = lerp(color, float3(color.r * 1.8, color.g * 0.6, color.b * 0.4), saturate(1.0 - redshift));
//     }
//     else
//     {
//         color = lerp(color, float3(color.r * 0.4, color.g * 0.6, color.b * 1.8), saturate(1.0 - redshift));
//     }
//     return color;
// }

// ------ Accretion Volumetric rendering helpers ------


float3 rotateY(float3 v, float angle)
{
    float s = sin(angle), c = cos(angle);
    return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

float remap(float x, float a, float b, float c, float d)
{
    float t = saturate((x - a) / max(b - a, 1e-6));
    return lerp(c, d, t);
}

float3 diskColorRamp(float t)
{
    // t = 0 outer (cool), t = 1 inner (hot)
    float3 cool = float3(0.95, 0.45, 0.25);
    float3 hot = float3(1.0, 0.98, 0.92);
    return lerp(cool, hot, saturate(t));
}

void sampelDiskAt(
    float3 p,
    float3 rayDirLocal,
    float M, float a,
    float time,
    out float rho,
    out float3 emissive
)
{
    // Cylindrical coords around +Y spin axis
    float2 xz = float2(p.x, p.z);
    float rperp = length(xz);
    float h = p.y;

    // Radial soft annulus
    float inner = smoothstep(ubo.diskRin, ubo.diskRin + ubo.diskEdgeK, rperp);
    float outer = 1.0 - smoothstep(ubo.diskRout - ubo.diskEdgeK, ubo.diskRout, rperp);
    float radialMask = saturate(inner * outer);

    // Vertical gaussian thickness
    float vertical = exp(-(h*h) / max(2.0 * ubo.diskH * ubo.diskH, 1e-6));

    // base density
    float base = ubo.diskDensity * radialMask * vertical;

    // azimuthal advection (simple keplerian omega ~ v/r)
    float rSafe = max(rperp, 1e-3);
    float vK = saturate(sqrt(M / rSafe)) * ubo.vScale;
    float omega = (rSafe > 0.0) ? (vK / rSafe) : 0.0;
    float advectAngle = omega * time;

    // sample pos for noise w mild warping
    float3 pn = rotateY(p, advectAngle);
    pn *= ubo.diskNoiseScale;
    // float nBase = fbm3(pn, 4, 2.03, 0.52);
    // float3 pw = pn + ubo.diskNoiseWarp * float3(
    //     fbm3(pn + 11.3, 3, 2.07, 0.55),
    //     fbm3(pn + 27.1, 3, 2.07, 0.55),
    //     fbm3(pn + 41.7, 3, 2.07, 0.55)
    // );
    // float nDetail = fbm3(pw, 3, 2.13, 0.5);

    float nBase = noiseTexture.SampleLevel(noiseSampler, pn, 0).r;
    float3 warpOffset = float3(
        noiseTexture.SampleLevel(noiseSampler, pn + 11.3, 0).r - 0.5,
        noiseTexture.SampleLevel(noiseSampler, pn + 27.1, 0).r - 0.5,
        noiseTexture.SampleLevel(noiseSampler, pn + 41.7, 0).r - 0.5
    );
    float3 pw = pn + ubo.diskNoiseWarp * warpOffset;
    float nDetail = noiseTexture.SampleLevel(noiseSampler, pw, 0).r;


    float n = lerp(nBase, nDetail, ubo.diskNoiseAmp);
    n = remap(n, 0.3, 0.8, 0.0, 1.0); // tighten contrast
    rho = max(0.0, base * n);

    // temperature color  by radius
    float tRad = 1.0 - saturate((rperp - ubo.diskRin) / max(ubo.diskRout - ubo.diskRin, 1e-3));
    tRad = saturate(tRad + ubo.temperatureBias * (tRad - 0.5));
    float3 baseCol = diskColorRamp(tRad);

    // doppler boosting (approximated by camera frame)
    float3 phiHat = (rperp > 1e-5) ? normalize(float3(-p.z, 0.0, p.x)) : float3(0, 0, 0);
    float3 v = vK * phiHat; // gas velocity
    float v2 = dot(v, v);
    float gamma = rsqrt(max(1.0 - v2, 1e-4));
    float cosTheta = dot(v, -rayDirLocal); // emission travels toward camera
    float D = 1.0 / (gamma * max(1.0 - cosTheta, 1e-4));

    // gravitational factor via lapse at this point (cheap)
    float r = length(p);
    float2 sph = cartesianToSpherical(p);
    KerrMetricComponents metric = computeKerrMetric(r, sph.x, a, M);
    float grav = 1.0 / max(metric.alpha, 1e-3);

    // emissive color (base -> doppler boost -> gravitational)
    emissive = baseCol * ubo.emissionScale;
    emissive *= pow(D, ubo.dopplerPower);
    emissive *= grav;
}

void sampleDiskImproved(
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
    float innerRadius = ubo.diskRin;
    float outerRadius = ubo.diskRout;
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

    // float angleOffset = theta + rotationSpeed * time;

    // float3 noisePos = float3(
    //     cos(angleOffset) * r / outerRadius,
    //     h / diskHeight,
    //     sin(angleOffset) * r / outerRadius
    // );
    // float noise1 = noiseTexture.SampleLevel(noiseSampler, noisePos * 2.0, 0).r;
    // float noise2 = noiseTexture.SampleLevel(noiseSampler, noisePos * 5.0, 0).r;
    // float noise3 = noiseTexture.SampleLevel(noiseSampler, noisePos * 10.0, 0).r;

    float turbulence = noise1 + 0.3 * noise2 + 0.1 * noise3;
    turbulence = turbulence / 1.4;

    // float turbulenceStrength = 0.3 + 0.7 * (r - innerRadius) / (outerRadius - innerRadius);
    // density = radialMask * verticalProfile * (0.7 + turbulenceStrength * turbulence);
    
    float structure = lerp(turbulence, spiralPattern * turbulence, 0.5);
    
    float radialDensityBoost = 1.0 + 2.0 * exp(-(r - innerRadius) / (3.0 * M));
    
    density = ubo.diskDensity * radialMask * verticalProfile * structure * radialDensityBoost;
    
    float clumpiness = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 0.5 + time * 0.1, 0).r;
    if (clumpiness > 0.7) {
        density *= 1.0 + (clumpiness - 0.7) * 3.0; // Boost density in clumps
    }

    // ----- color/emission calculation -----

    float temperatureParam = 1.0 - saturate((r - innerRadius) / (outerRadius - innerRadius));
    temperatureParam = pow(temperatureParam, 2.0);

    float3 coolColor = float3(0.8, 0.3, 0.1);     // Red-orange
    float3 warmColor = float3(1.0, 0.7, 0.3);     // Orange-yellow
    float3 hotColor = float3(1.0, 0.95, 0.8);     // Yellow-white
    float3 veryHotColor = float3(0.8, 0.85, 1.0); // Blue-white

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

    float dopplerBoost = pow(1.0 / max(dopplerFactor, 0.1), ubo.dopplerPower);  // Relativistic beaming

    if (cosAngle > 0) {
        // Moving toward observer - blueshift
        diskColor = lerp(diskColor, diskColor * float3(0.7, 0.85, 1.2), cosAngle * 0.3);
        emissive = diskColor * (1.0 + cosAngle * 0.5); // Brighter
    } else {
        // Moving away - redshift
        diskColor = lerp(diskColor, diskColor * float3(1.2, 0.85, 0.7), -cosAngle * 0.3);
        emissive = diskColor * (1.0 + cosAngle * 0.3); // Dimmer
    }

    // ----- gravitational effects -----

    float gravitationalRedshift = sqrt(1.0 - 2.0 * M / max(r, 2.0 * M));
    float compressionBoost = 1.0 + 1.0 * exp(-r / (5.0 * M));
    
    // Final emission with all effects
    emissive *= ubo.emissionScale * density * dopplerBoost * compressionBoost * gravitationalRedshift;
    
    // Hot spots and flares
    float flareNoise = noiseTexture.SampleLevel(noiseSampler, spiralCoords * 0.3 + time * 0.5, 0).r;
    if (flareNoise > 0.85 && temperatureParam > 0.6) {
        emissive *= 1.0 + (flareNoise - 0.85) * 5.0;
    }
}

float computeTransmittanceImproved(
    float3 position,
    float3 lightDir,
    float stepSize,
    int steps,
    float M,
    float a,
    float time,
    Texture3D<float4> noiseTexture,
    SamplerState noiseSampler
)
{
    float tau = 0.0;
    
    for (int i = 0; i < steps; i++) {
        float3 samplePos = position + lightDir * (float(i) * stepSize);
        
        float density;
        float3 emissive;  // Not used here
        sampleDiskImproved(samplePos, -lightDir, M, a, time, noiseTexture, noiseSampler, density, emissive);
        
        tau += density * stepSize;
    }
    
    return exp(-tau);
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

    float2 noiseCoord = float2(pixelCoords) / 64.0; //  64x64 blue noise texture
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
                        sampleDiskImproved(
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
                                // Linear approximation for small extinction
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
            finalColor = L.bgr;
        }
    }
    else
    {
        float3 bg = sampleBackground(rayState.direction);
        if (ubo.backgroundType == 1)
        {
            float exposure = ubo.exposure; // default 1.0
            float gamma = ubo.gamma; // default 2.2
            
            // bg = reinhardToneMapping(bg, exposure);
            // bg = exposureToneMapping(bg, exposure);
            bg = ACESFilm(bg * exposure);
            bg = pow(bg, float3(1.0 / gamma));
        }
        
        finalColor = L.bgr + T * bg;
    }

    storageImage[pixelCoords] = float4(finalColor, 1.0);
}