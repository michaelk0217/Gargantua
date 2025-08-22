#pragma once


#include <vulkan/vulkan.h>
#include "VulkanStructures.h"
class Window;
class VulkanDevice;
class VulkanSwapchain;

class UIOverlay
{
public:
	UIOverlay(Window& window, VulkanDevice& device, VulkanSwapchain& swapchain);
	~UIOverlay();

	UIOverlay(const UIOverlay&) = delete;
	UIOverlay& operator=(const UIOverlay&) = delete;

	void newFrame();
	void buildUI(UIPacket& uiPacket);
	void render(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height);

	static void update_frame_history(std::vector<float>& frame_history, float framerate);
private:
	VulkanDevice& m_device;
	VulkanSwapchain& m_swapchain;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

};

