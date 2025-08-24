#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <vulkan/vulkan.h>
#include <array>
#include <string>
#include <memory>

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	//glm::vec2 texCoord;
	//glm::vec3 inNormal;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		/*attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, inNormal);*/

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && color == other.color /*&& texCoord == other.texCoord && inNormal == other.inNormal*/;
	}
};

namespace std {
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return  ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1)/* ^
				(hash<glm::vec2>()(vertex.texCoord) << 1) ^
				(hash<glm::vec3>()(vertex.inNormal) << 2)*/;
		}
	};
}


struct ShaderData {
	alignas(16) glm::mat4 projectionMatrix;
	alignas(16) glm::mat4 modelMatrix;
	alignas(16) glm::mat4 viewMatrix;

	alignas(16) glm::mat4 inverseProjectionMatrix;
	alignas(16) glm::mat4 inverseViewMatrix;
	alignas(16) glm::vec3 cameraPosition;

	alignas(4) float time;
	alignas(4) int backgroundType = 0;

	alignas(4) float exposure = 1.0f;
	alignas(4) float gamma = 2.2f;

	alignas(4) float blackHoleMass = 1.0f;
	alignas(4) float blackHoleSpin = 0.6f;
	alignas(4) int maxSteps = 100;
	alignas(4) float stepSize = 0.2f;
	alignas(4) int geodesicType = 0;

	// accretion disk params
	alignas(4) int diskEnable = 0;
	alignas(4) int volSubsteps = 6;
	alignas(4) float dsVolScale = 1.0f;
	alignas(4) float sigmaT = 2.0f;

	// parameters scaled by blackholemass default
	alignas(4) float diskRin = 3.0f * 1.0f; // inner radius ( 3.0 * M )
	alignas(4) float diskRout = 15.0f * 1.0f; // outer radius 
	alignas(4) float diskH = 0.1f * 1.0f; // vertical scale height
	alignas(4) float diskEdgeK = 0.8f * 0.1f; // softness of inner/outer edge

	alignas(4) float diskDensity = 0.3f;
	alignas(4) float diskNoiseAmp = 0.6f;
	alignas(4) float diskNoiseScale = 2.5f;
	alignas(4)float diskNoiseWarp = 0.5f;

	alignas(4) float emissionScale = 3.5f; // overall disk brightness
	alignas(4) float vScale = 0.6f; // Keplerian seed multiplier
	alignas(4) float dopplerPower = 0.5f;
	alignas(4) float temperatureBias = 0.35f;

};

struct NoiseUboParams
{
	alignas(4) int size;
	alignas(4) float scale;
	alignas(4) float boost;
	alignas(4) int octaves;
	alignas(4) int tileSize;
	alignas(4) float lacunarity;
	alignas(4) float gain;
	alignas(4) float time;
};

struct UIPacket {
	float& deltaTime;
	std::vector<float>& frameHistory;
	glm::vec3& cameraDirection;

	float& blackHoleMass;
	float& blackHoleSpin;
	int& maxSteps;
	float& stepSize;
	int& backgroundType;
	int& geodesicType;

	bool& diskEnable;
};