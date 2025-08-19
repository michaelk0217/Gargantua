#include "VulkanSwapchain.h"

VulkanSwapchain::VulkanSwapchain(VkInstance instance, VkSurfaceKHR surface, VkDevice device, VkPhysicalDevice physicalDevice, GLFWwindow* window)
{
	this->instance = instance;
	this->surface = surface;
	this->physicalDevice = physicalDevice;
	this->device = device;
	this->window = window;
}

VulkanSwapchain::~VulkanSwapchain()
{
	cleanup();
}

void VulkanSwapchain::create(uint32_t& width, uint32_t& height, bool vsync, bool fullscreen)
{
	assert(physicalDevice);
	assert(device);
	assert(instance);
	assert(surface);

	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	this->colorFormat = surfaceFormat.format;
	this->colorSpace = surfaceFormat.colorSpace;

	// Store the current swap chain handle so we can use it later on to ease up recreation
	VkSwapchainKHR oldSwapchain = swapChain;

	// Get physical device surface properties and formats
	VkSurfaceCapabilitiesKHR surfCaps;
	VK_CHECK_RESULT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfCaps));

	VkExtent2D swapchainExtent = {};
	//// If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
	//if (surfCaps.currentExtent.width == (uint32_t)-1)
	//{
	//	// If the surface size is undefined, the size is set to the size of the images requested
	//	swapchainExtent.width = width;
	//	swapchainExtent.height = height;
	//}
	//else
	//{
	//	// If the surface size is defined, the swap chain size must match
	//	swapchainExtent = surfCaps.currentExtent;
	//	width = surfCaps.currentExtent.width;
	//	height = surfCaps.currentExtent.height;
	//}

	if (surfCaps.currentExtent.width == (uint32_t)-1)
	{
		int fbWidth, fbHeight;
		glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

		swapchainExtent = {
			static_cast<uint32_t>(fbWidth),
			static_cast<uint32_t>(fbHeight)
		};
		swapchainExtent.width = std::clamp(swapchainExtent.width, surfCaps.minImageExtent.width, surfCaps.maxImageExtent.width);
		swapchainExtent.height = std::clamp(swapchainExtent.height, surfCaps.minImageExtent.height, surfCaps.maxImageExtent.height);
	}
	else
	{
		swapchainExtent = surfCaps.currentExtent;
	}
	width = swapchainExtent.width;
	height = swapchainExtent.height;

	extent = swapchainExtent;

	// Select a present mode for the swapchain
	uint32_t presentModeCount;
	VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr));
	assert(presentModeCount > 0);

	std::vector<VkPresentModeKHR> presentModes(presentModeCount);
	VK_CHECK_RESULT(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data()));

	// The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
	// This mode waits for the vertical blank ("v-sync")
	VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;

	if (!vsync)
	{
		for (size_t i = 0; i < presentModeCount; i++)
		{
			if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
				break;
			}
			if (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
			}
		}
	}

	// Determine the number of images
	uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
	if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount))
	{
		desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
	}

	// Find the transformation of the surface
	VkSurfaceTransformFlagsKHR preTransform;
	if (surfCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		// We prefer a non-rotated transform
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		preTransform = surfCaps.currentTransform;
	}

	// Find a supported composite alpha format (not all devices support alpha opaque)
	VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	// Simply select the first composite alpha format available
	std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
		VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	};
	for (auto& compositeAlphaFlag : compositeAlphaFlags)
	{
		if (surfCaps.supportedCompositeAlpha & compositeAlphaFlag)
		{
			compositeAlpha = compositeAlphaFlag;
			break;
		}
	}

	VkSwapchainCreateInfoKHR swapchainCI = {};
	swapchainCI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCI.surface = surface;
	swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
	swapchainCI.imageFormat = colorFormat;
	swapchainCI.imageColorSpace = colorSpace;
	swapchainCI.imageExtent = { swapchainExtent.width, swapchainExtent.height };
	swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchainCI.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
	swapchainCI.imageArrayLayers = 1;
	swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCI.queueFamilyIndexCount = 0;
	swapchainCI.presentMode = swapchainPresentMode;
	// Setting oldSwapChain to the saved handle of the previous swapchain aids in resource reuse and makes sure that we can still present already acquired images
	swapchainCI.oldSwapchain = oldSwapchain;
	// Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
	swapchainCI.clipped = VK_TRUE;
	swapchainCI.compositeAlpha = compositeAlpha;

	// Enable transfer source on swap chain images if supported
	if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
	{
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}

	// Enable transfer destination on swap chain images if supported
	if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
	{
		swapchainCI.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	VK_CHECK_RESULT(vkCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapChain));

	// If an existing swap chain is re-created, destroy the old swap chain and the resources owned by the application (image views, images are owned by the swap chain)
	if (oldSwapchain != VK_NULL_HANDLE)
	{
		for (auto i = 0; i < images.size(); i++)
		{
			vkDestroyImageView(device, imageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
	}

	VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr));

	// Get the swap chain images
	images.resize(imageCount);
	VK_CHECK_RESULT(vkGetSwapchainImagesKHR(device, swapChain, &imageCount, images.data()));

	// Get the swap chain buffers containing the image and imageview
	imageViews.resize(imageCount);
	for (auto i = 0; i < images.size(); i++)
	{
		VkImageViewCreateInfo colorAttachmentView = {};
		colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		colorAttachmentView.pNext = nullptr;
		colorAttachmentView.format = colorFormat;
		colorAttachmentView.components = {
			VK_COMPONENT_SWIZZLE_R,
			VK_COMPONENT_SWIZZLE_G,
			VK_COMPONENT_SWIZZLE_B,
			VK_COMPONENT_SWIZZLE_A
		};
		colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorAttachmentView.subresourceRange.baseMipLevel = 0;
		colorAttachmentView.subresourceRange.levelCount = 1;
		colorAttachmentView.subresourceRange.baseArrayLayer = 0;
		colorAttachmentView.subresourceRange.layerCount = 1;
		colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorAttachmentView.flags = 0;
		colorAttachmentView.image = images[i];
		VK_CHECK_RESULT(vkCreateImageView(device, &colorAttachmentView, nullptr, &imageViews[i]));
	}
}

VkResult VulkanSwapchain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t& imageIndex)
{
	// By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
	// With that we don't have to handle VK_NOT_READY
	return vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, presentCompleteSemaphore, (VkFence)nullptr, &imageIndex);
}

VkResult VulkanSwapchain::queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore)
{
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = NULL;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapChain;
	presentInfo.pImageIndices = &imageIndex;
	// Check if a wait semaphore has been specified to wait for before presenting the image
	if (waitSemaphore != VK_NULL_HANDLE)
	{
		presentInfo.pWaitSemaphores = &waitSemaphore;
		presentInfo.waitSemaphoreCount = 1;
	}
	return vkQueuePresentKHR(queue, &presentInfo);
}

void VulkanSwapchain::cleanup()
{
	if (swapChain != VK_NULL_HANDLE) {
		for (auto i = 0; i < images.size(); i++) {
			vkDestroyImageView(device, imageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}
	swapChain = VK_NULL_HANDLE;
}

SwapChainSupportDetails VulkanSwapchain::querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

VkSurfaceFormatKHR VulkanSwapchain::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return availableFormat;
		}
	}

	return availableFormats[0];
}


