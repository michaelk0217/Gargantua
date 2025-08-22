// Uniform buffer structure
struct ShaderData
{
    column_major float4x4 projectionMatrix;
    column_major float4x4 modelMatrix;
    column_major float4x4 viewMatrix;
    column_major float4x4 inverseProjectionMatrix;
    column_major float4x4 inverseViewMatrix;
    float3 cameraPosition;
};

// Uniform buffer binding
[[vk::binding(0, 0)]]
ConstantBuffer<ShaderData> ubo;

// Vertex shader input structure
struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

// Vertex shader output / Fragment shader input structure
struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

// Vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;

    float4 worldPos = mul(ubo.modelMatrix, float4(input.position, 1.0));
    float4 viewPos = mul(ubo.viewMatrix, worldPos);
    output.position = mul(ubo.projectionMatrix, viewPos);

    output.color = input.color;
    
    return output;
}

// Fragment shader entry point
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}