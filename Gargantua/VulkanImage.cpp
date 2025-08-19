#include "VulkanImage.h"
#include "VulkanTools.h"


/**
* Creates Image, Imageview, and memory.
* Must have a complete imageInfo and viewInfo (createInfo) before calling this function.
*/
void vks::Image::createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties)
{
	this->device = device;
	VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &image));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, image, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = vks::tools::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &memory));
	
	vkBindImageMemory(device, image, memory, 0);

	viewInfo.image = image;
	VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &imageView));
}

void vks::Image::destroy()
{
	if (image != VK_NULL_HANDLE)
	{
		vkDestroyImage(device, image, nullptr);
		image = VK_NULL_HANDLE;
	}
	if (imageView != VK_NULL_HANDLE)
	{
		vkDestroyImageView(device, imageView, nullptr);
		imageView = VK_NULL_HANDLE;
	}
	if (memory != VK_NULL_HANDLE)
	{
		vkFreeMemory(device, memory, nullptr);
		memory = VK_NULL_HANDLE;
	}
}