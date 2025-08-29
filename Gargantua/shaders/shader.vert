#version 450
// #extension GL_ARB_separate_shader_objects : enable

// Uniform buffer object
layout(binding = 0) uniform UniformBufferObject {
    mat4 proj;
    mat4 model;
    mat4 view;
} ubo;

// Vertex input
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

// Vertex output
layout(location = 0) out vec3 outColor;

void main() {
    vec4 worldPos = ubo.model * vec4(position, 1.0);
    vec4 viewPos = ubo.view * worldPos;
    gl_Position = ubo.proj * viewPos;
    
    outColor = color;
}