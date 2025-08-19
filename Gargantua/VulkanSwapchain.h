#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <vector>

#include <vulkan/vulkan.h>
#include "VulkanTools.h"


struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class VulkanSwapchain
{
private:
	VkInstance instance{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	GLFWwindow* window = nullptr;

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

public:
	VkExtent2D extent{};
	VkFormat colorFormat{};
	VkColorSpaceKHR colorSpace{};
	VkSwapchainKHR swapChain{ VK_NULL_HANDLE };
	std::vector<VkImage> images{};
	std::vector<VkImageView> imageViews{};
	uint32_t queueNodeIndex{ UINT32_MAX };
	uint32_t imageCount{ 0 };

	VulkanSwapchain(VkInstance instance, VkSurfaceKHR surface, VkDevice device, VkPhysicalDevice physicalDevice, GLFWwindow* window);
	~VulkanSwapchain();

	void create(uint32_t& width, uint32_t& height, bool vsync = false, bool fullscreen = false);
	VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t& imageIndex);
	VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);
	void cleanup();
	
	static SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
};

