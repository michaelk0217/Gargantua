#version 450
// #extension GL_ARB_separate_shader_objects : enable

// Fragment input
layout(location = 0) in vec3 inColor;

// Fragment output
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor, 1.0);
}