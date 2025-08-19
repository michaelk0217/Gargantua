// Uniform buffer structure
struct UniformBufferObject
{
    column_major float4x4 proj;
    column_major float4x4 model;
    column_major float4x4 view;
}

// Uniform buffer binding
[[vk::binding(0, 0)]]
ConstantBuffer<UniformBufferObject> ubo;

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
    
    // Calculate the final position in clip space
    float4 worldPos = mul(ubo.model, float4(input.position, 1.0));
    float4 viewPos = mul(ubo.view, worldPos);
    output.position = mul(ubo.proj, viewPos);

    // Pass the input color to the fragment shader
    output.color = input.color;
    
    return output;
}

// Fragment shader entry point
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}