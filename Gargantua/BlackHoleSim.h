#pragma once

#include <memory>

#include <vulkan/vulkan.h>
#include <slang/slang.h>
#include "VulkanDevice.h"
#include "VulkanSwapchain.h"
#include "VulkanStructures.h"
#include "VulkanBuffer.h"
#include "VulkanImage.h"

#include "Window.h"
#include "Camera.hpp"


class BlackHoleSim
{

public:
	void run();
private:
	uint32_t width = 2560;
	uint32_t height = 1440;
	const std::string windowTitle = "Gargantua";
	bool resized = false;
	uint32_t currentFrame = 0;
	static const uint32_t MAX_CONCURRENT_FRAMES = 2;

	// slang global session
	slang::IGlobalSession* slangGlobalSession;

	std::unique_ptr<Window> window;
	std::unique_ptr<Camera> camera;
	std::unique_ptr<VulkanDevice> device;
	std::unique_ptr<VulkanSwapchain> swapchain;

	std::vector<VkCommandBuffer> frameCommandBuffers;
	//std::vector<VkCommandBuffer> computeCommandBuffers;

	// ------ main functions ------
	void initGlfwWindow();
	void initVulkan();
	void mainLoop();
	void cleanUp();
	// ----------------------------
	void drawFrame();
	void windowResize();


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

	//void createDepthResources();
	//vks::Image depthImage;

	//void createVertexBuffer();
	//vks::Buffer vertexBuffer;
	//void createIndexBuffer();
	//vks::Buffer indexBuffer;



	// triangle test
	/*const std::vector<Vertex> vertices{
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
			{ {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
	};
	std::vector<uint32_t> indices{ 0, 1, 2 };*/
};

