// ================== Post Processing Vertex ==================
struct PPVSout
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

PPVSout PPVertMain(uint vertexId : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };

    PPVSout output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.uv = (positions[vertexId] * 0.5) + 0.5; 
    return output;
}

// ================== Post Processing Fragment ==================
Texture2D hdrScene        : register(t0);
SamplerState hdrSampler   : register(s0);

cbuffer UBO : register(b1)
{
    float exposure;
    float gamma;
};

float3 saturate(float3 x)
{
    return clamp(x, 0.0, 1.0);
}

float3 ACESFilm(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float4 PPFragMain(PPVSout input) : SV_Target
{
    float2 texSize;
    hdrScene.GetDimensions(texSize.x, texSize.y);

    float2 uv = input.uv;
    float3 color = hdrScene.Sample(hdrSampler, uv).rgb;

    color *= exposure;

    color = ACESFilm(color);

    color = pow(color, 1.0 / gamma);

    return float4(color, 1.0);
}
