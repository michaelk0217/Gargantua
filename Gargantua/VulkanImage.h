#pragma once

#include <vulkan/vulkan.h>

namespace vks
{
	struct Image
	{
		VkDevice device;
		VkImage image = VK_NULL_HANDLE;
		VkImageView imageView = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;

		VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

		VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };

		//VkImageType imageType;
		//VkExtent3D extent3d;
		//uint32_t mipLevels;
		//uint32_t arrayLayers;
		//VkFormat format;
		//VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
		//VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		//VkImageUsageFlags usage;
		//VkSampleCountFlagBits samples;
		//VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		//VkImageCreateFlags flags = 0;

		void createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties);
		void destroy();
	};
}
