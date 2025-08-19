#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <assert.h>

#include <vulkan/vulkan.h>


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	// Select prefix depending on flags passed to the callback
	std::string prefix;

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
	{
		prefix = "VERBOSE: ";
	}
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		prefix = "INFO: ";
	}
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		prefix = "WARNING: ";
	}
	else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		prefix = "ERROR: ";
	}

	//Display message to defualt output (console/logcat)
	std::stringstream debugMessage;
	if (pCallbackData->pMessageIdName)
	{
		debugMessage << prefix << "[" << pCallbackData->messageIdNumber << "][" << pCallbackData->pMessageIdName << "] : " << pCallbackData->pMessage;
	}
	else
	{
		debugMessage << prefix << "[" << pCallbackData->messageIdNumber << "] : " << pCallbackData->pMessage;
	}

	if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		std::cerr << debugMessage.str() << "\n\n";
	}
	else
	{
		std::cout << debugMessage.str() << "\n\n";
	}
	fflush(stdout);

	// The return value of this callback controls whether the Vulkan call that caused the validation message will be aborted or not
	// We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message to abort
	// If you instead want to have calls abort, pass in VK_TRUE and the function will return VK_ERROR_VALIDATION_FAILED_EXT
	return VK_FALSE;
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> computeFamily;
	std::optional<uint32_t> transferFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value() && computeFamily.has_value() && transferFamily.has_value() && presentFamily.has_value();
	}
};

class VulkanDevice
{

public:
	VulkanDevice(GLFWwindow* window);
	~VulkanDevice();

	GLFWwindow* window;

	VkInstance instance = VK_NULL_HANDLE;
	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

	VkSurfaceKHR surface;

	VkDevice logicalDevice;
	VkPhysicalDevice physicalDevice;
	VkQueue graphicsQueue;
	VkQueue computeQueue;
	VkQueue transferQueue;
	VkQueue presentQueue;
	QueueFamilyIndices familyIndices;

	VkCommandPool graphicsCommandPool;
	VkCommandPool computeCommandPool;
	VkCommandPool transferCommandPool;

	// ----- Vulkan Command Pool / Buffer -----
	static VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	static std::vector<VkCommandBuffer> createCommandBuffers(VkDevice device, VkCommandBufferLevel level, VkCommandPool pool, uint32_t count);

	static VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandBufferLevel level, VkCommandPool pool, bool begin);
private:

	// ----- Vulkan Instance & Validation Layer -----
	void createInstance();
	void setupDebugging();
	void freeDebugCallback();
	bool checkValidationLayerSupport() const;
	std::vector<const char*> getRequiredExtensions() const;
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
	PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

	// ----- Vulkan Surface -----
	void createSurface(GLFWwindow* window);

	// ----- Vulkan Devices -----
	void pickPhysicalDevice();
	void createLogicalDevice();
	bool isDeviceSuitable(VkPhysicalDevice physicalDevice, VkSurfaceKHR vkSurface);
	bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice);

	static QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
	static uint32_t getQueueFamilyIndex(VkQueueFlags queueFlags, std::vector<VkQueueFamilyProperties> familyProperties);

	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		"VK_KHR_dynamic_rendering",
		"VK_KHR_synchronization2"
	};

	
};

