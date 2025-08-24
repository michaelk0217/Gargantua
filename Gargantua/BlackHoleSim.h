#pragma once

#include <memory>
#include <chrono>

#include <vulkan/vulkan.h>
#include <slang/slang.h>
#include "VulkanDevice.h"
#include "VulkanSwapchain.h"
#include "VulkanStructures.h"
#include "VulkanBuffer.h"
#include "VulkanImage.h"

#include "Window.h"
#include "Camera.hpp"
#include "UIOverlay.h"

class BlackHoleSim
{

public:
	void run();
private:
	uint32_t width = 3840;
	uint32_t height = 2160;
	const std::string windowTitle = "Gargantua";
	bool resized = false;
	uint32_t currentFrame = 0;
	static const uint32_t MAX_CONCURRENT_FRAMES = 2;
	float totalElapsedTime = 0.0f;

	std::vector<float> frame_history;

	// slang global session
	slang::IGlobalSession* slangGlobalSession;

	std::unique_ptr<Window> window;
	std::unique_ptr<Camera> camera;
	std::unique_ptr<VulkanDevice> device;
	std::unique_ptr<VulkanSwapchain> swapchain;

	std::vector<VkCommandBuffer> frameCommandBuffers;
	//std::vector<VkCommandBuffer> computeCommandBuffers;

	std::unique_ptr<UIOverlay> uiOverlay;

	// ------ main functions ------
	void initGlfwWindow();
	void initVulkan();
	void mainLoop();
	void cleanUp();
	// ----------------------------
	void drawFrame();
	void windowResize();
	void processInput(float deltaTime);


	// ------ SYNC OBJECTS ------
	void createSyncPrimitives();
	std::vector<VkSemaphore> presentCompleteSemaphores{};
	std::vector<VkSemaphore> renderCompleteSemaphores{};
	std::array<VkFence, MAX_CONCURRENT_FRAMES> waitFences{};

	// ------ Compute Pipeline -------
	void createComputePipeline();
	void updateComputeDescriptorSets();
	VkPipeline computePipeline;
	VkPipelineLayout computePipelineLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;
	std::vector<VkDescriptorSet> computeDescriptorSets;
	void cleanupComputePipeline();

	void createComputeStorageImage();
	vks::Image computeStorageImage;

	// ------ Graphics Render Pipeline ------
	/*void createGraphicsPipeline();
	VkDescriptorSetLayout graphicsDescriptorSetLayout;
	VkPipelineLayout graphicsPipelineLayout;
	VkPipeline graphicsPipeline;
	void updateGraphicsDescriptorSet(uint32_t currentFrameIndex, VkBuffer uboBuffer);
	std::vector<VkDescriptorSet> graphicsDescriptorSets;*/
	void initializeFrameUbo();
	std::vector<vks::Buffer> frameUBO;

	// ------ Resources ------
	static void allocateDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, uint32_t setCount, std::vector<VkDescriptorSet>& descriptorSets);

	void createDescriptorPool();
	VkDescriptorPool descriptorPool;

	/*void createDepthResources();
	vks::Image depthImage;*/

	/*void createVertexBuffer();
	vks::Buffer vertexBuffer;
	void createIndexBuffer();
	vks::Buffer indexBuffer;*/

	void loadEnvironmentTextures();
	void createDummyHDRTexture(vks::Image& texture);
	void createEnvironmentSampler();
	vks::Image spheremapTexture;
	VkSampler environmentSampler;

	// noise generationpipeline
	void createNoiseResources(int size, VkFormat format = VK_FORMAT_R16_SFLOAT);
	void destroyNoiseResources();
	vks::Image noiseTexture3D;
	VkSampler noiseSampler;
	vks::Buffer noiseUBO;
	VkDescriptorSetLayout noiseDescriptorSetLayout;
	VkPipeline noiseComputePipeline;
	VkPipelineLayout noiseComputePipelineLayout;
	VkDescriptorSet noiseComputeDescriptorSet;
	void generateNoise3D(int size);
	const int NOISE_SIZE = 64;


	// UI Controlled Parameters
	float blackHoleMass = 1.0f;
	float blackHoleSpin = 0.6f;
	int maxSteps = 100;
	float stepSize = 0.2f;
	int backgroundType = 0;
	int geodesicType = 0;

	bool diskEnable = false;
};

