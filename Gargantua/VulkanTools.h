/* This code is from SaschaWillems Vulkan/base/VulkanTools.h */
#pragma once

#include <vulkan/vulkan.h>
#include <slang/slang.h>
#include <slang/slang-com-ptr.h>
#include <slang/slang-com-helper.h>
#include "VulkanInitializers.hpp"

#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#define VK_FLAGS_NONE 0
#define DEFAULT_FENCE_TIMEOUT 100000000000 // in nanoseconds

#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		std::cout << "Fatal : VkResult is \"" << vks::tools::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
		assert(res == VK_SUCCESS);																		\
	}																									\
}

const std::string getAssetPath();
const std::string getShaderBasePath();

namespace vks
{
	namespace tools
	{
		/** @brief Setting this path changes the place where the samples looks for assets and shaders */
		//extern std::string resourcePath;

		/** @brief Disable message boxes on fatal errors */
		extern bool errorModeSilent;

		/** @brief Returns an error code as a string */
		std::string errorString(VkResult errorCode);

		/** @brief Returns the device type as a string */
		std::string physicalDeviceTypeString(VkPhysicalDeviceType type);

		uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

		// Selected a suitable supported depth format starting with 32 bit down to 16 bit
		// Returns false if none of the depth formats in the list is supported by the device
		VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat);
		// Same as getSupportedDepthFormat but will only select formats that also have stencil
		VkBool32 getSupportedDepthStencilFormat(VkPhysicalDevice physicalDevice, VkFormat* depthStencilFormat);

		// Returns true a given format support LINEAR filtering
		VkBool32 formatIsFilterable(VkPhysicalDevice physicalDevice, VkFormat format, VkImageTiling tiling);
		// Returns true if a given format has a stencil part
		VkBool32 formatHasStencil(VkFormat format);

		// Put an image memory barrier for setting an image layout on the sub resource into the given command buffer
		void setImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkImageSubresourceRange subresourceRange,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		// Uses a fixed sub resource layout with first mip level and layer
		void setImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageAspectFlags aspectMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		/** @brief Insert an image memory barrier into the command buffer */
		void insertImageMemoryBarrier(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkAccessFlags srcAccessMask,
			VkAccessFlags dstAccessMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask,
			VkImageSubresourceRange subresourceRange);

		/** @brief Insert a buffer memory barrier into the command buffer */
		void insertBufferMemoryBarrier(
			VkCommandBuffer cmdbuffer,
			VkAccessFlags srcAccessMask,
			VkAccessFlags dstAccessMask,
			VkBuffer buffer,
			VkDeviceSize offset,
			VkDeviceSize size,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask);

		/** @brief insert Sync2 Memory barrier into command buffer */
		void insertMemoryBarrier2(
			VkCommandBuffer cmdbuffer,
			VkAccessFlags2 srcAccessMask,
			VkAccessFlags2 dstAccessMask,
			VkPipelineStageFlags2 srcStageMask,
			VkPipelineStageFlags2 dstStageMask
		);

		// Display error message and exit on fatal error
		void exitFatal(const std::string& message, int32_t exitCode);
		void exitFatal(const std::string& message, VkResult resultCode);

		// Load a SPIR-V shader (binary)
		VkShaderModule loadShader(const char* fileName, VkDevice device);

		// reads file into string
		std::string readFile(const char* filePath);

		// Load slang shader
		VkShaderModule loadSlangShader(VkDevice device, slang::IGlobalSession* slangGlobalSession, const char* shaderPath, const char* entryPointName);
		void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob);


		VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);
		void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkDevice device, VkQueue queue, VkCommandPool commandPool);

		/** @brief Checks if a file exists */
		bool fileExists(const std::string& filename);

		uint32_t alignedSize(uint32_t value, uint32_t alignment);
		VkDeviceSize alignedVkSize(VkDeviceSize value, VkDeviceSize alignment);
	}
}

