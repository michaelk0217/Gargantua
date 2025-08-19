#include "VulkanDevice.h"
#include "VulkanTools.h"
#include "VulkanSwapchain.h"
VulkanDevice::VulkanDevice(GLFWwindow* window)
{
	this->window = window;
	createInstance();
	setupDebugging();
	createSurface(window);
	pickPhysicalDevice();
	createLogicalDevice();
	graphicsCommandPool = createCommandPool(logicalDevice, familyIndices.graphicsFamily.value()); // also use for present queue
	computeCommandPool = createCommandPool(logicalDevice, familyIndices.computeFamily.value());
	transferCommandPool = createCommandPool(logicalDevice, familyIndices.transferFamily.value());


}

VulkanDevice::~VulkanDevice()
{
	vkDestroyCommandPool(logicalDevice, transferCommandPool, nullptr);
	vkDestroyCommandPool(logicalDevice, computeCommandPool, nullptr);
	vkDestroyCommandPool(logicalDevice, graphicsCommandPool, nullptr);
	vkDestroyDevice(logicalDevice, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	freeDebugCallback();
	vkDestroyInstance(instance, nullptr);
}

void VulkanDevice::createInstance()
{
	if (enableValidationLayers && !checkValidationLayerSupport())
	{
		throw std::runtime_error("Validation layer not supported");
	}

	VkApplicationInfo appInfo{};

	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "FluidSim";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 4, 0);
	appInfo.pEngineName = "FluidSim";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 4, 0);
	appInfo.apiVersion = VK_API_VERSION_1_4;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	auto extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugMessengerCI{};
	VkValidationFeaturesEXT validation_features{ VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
	VkLayerSettingEXT layerSetting{};

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		populateDebugMessengerCreateInfo(debugMessengerCI);

		std::vector<VkValidationFeatureEnableEXT> validation_feature_enables = {
			VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
		};
		validation_features.enabledValidationFeatureCount = static_cast<uint32_t>(validation_feature_enables.size());
		validation_features.pEnabledValidationFeatures = validation_feature_enables.data();

		layerSetting.pLayerName = "VK_LAYER_KHRONOS_validation";
		layerSetting.pSettingName = "enables";
		layerSetting.type = VK_LAYER_SETTING_TYPE_STRING_EXT;
		layerSetting.valueCount = 1;
		static const char* layerEnables = "VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT";
		layerSetting.pValues = &layerEnables;
		VkLayerSettingsCreateInfoEXT layerSettingsCI{};
		layerSettingsCI.sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
		layerSettingsCI.pNext = nullptr;
		layerSettingsCI.settingCount = 1;
		layerSettingsCI.pSettings = &layerSetting;

		validation_features.pNext = &layerSettingsCI;
		//debugMessengerCI.pNext = &validation_features;
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugMessengerCI;
	}
	else
	{
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = nullptr;
		//createInfo.pNext = &validation_features;
	}

	VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance));
}

void VulkanDevice::setupDebugging()
{
	vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

	VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCI{};
	populateDebugMessengerCreateInfo(debugUtilsMessengerCI);
	VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerCI, nullptr, &debugMessenger);
	assert(result == VK_SUCCESS);
}

void VulkanDevice::freeDebugCallback()
{
	if (debugMessenger != VK_NULL_HANDLE)
	{
		vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		debugMessenger = VK_NULL_HANDLE;
	}
}

bool VulkanDevice::checkValidationLayerSupport() const
{
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers)
	{
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
		{
			return false;
		}
	}

	return true;
}

std::vector<const char*> VulkanDevice::getRequiredExtensions() const
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
	return std::vector<const char*>();
}

void VulkanDevice::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
	createInfo.pUserData = nullptr;
}

void VulkanDevice::createSurface(GLFWwindow* window)
{
	VK_CHECK_RESULT(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

void VulkanDevice::pickPhysicalDevice()
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	assert(deviceCount > 0);
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	// Pick the first physical device
	for (const auto& physDevice : devices)
	{
		if (isDeviceSuitable(physDevice, surface))
		{
			this->physicalDevice = physDevice;
			break;
		}
	}
	if (physicalDevice == VK_NULL_HANDLE)
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}

void VulkanDevice::createLogicalDevice()
{
	familyIndices = findQueueFamilies(physicalDevice, surface);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { familyIndices.graphicsFamily.value(), familyIndices.computeFamily.value(), familyIndices.transferFamily.value(), familyIndices.presentFamily.value() };

	float queuePriority = 1.0f;

	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo queueCI{};
		queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCI.queueFamilyIndex = queueFamily;
		queueCI.queueCount = 1;
		queueCI.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCI);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.tessellationShader = VK_TRUE;
	deviceFeatures.fillModeNonSolid = VK_TRUE;
	deviceFeatures.wideLines = VK_TRUE;

	VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature{};
	dynamic_rendering_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
	dynamic_rendering_feature.dynamicRendering = VK_TRUE;

	VkPhysicalDeviceVulkan13Features vk13Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
	vk13Features.dynamicRendering = VK_TRUE;
	vk13Features.synchronization2 = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();
	//createInfo.pNext = &dynamic_rendering_feature;
	createInfo.pNext = &vk13Features;

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}

	VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice));

	vkGetDeviceQueue(logicalDevice, familyIndices.graphicsFamily.value(), 0, &graphicsQueue);
	vkGetDeviceQueue(logicalDevice, familyIndices.computeFamily.value(), 0, &computeQueue);
	vkGetDeviceQueue(logicalDevice, familyIndices.transferFamily.value(), 0, &transferQueue);
	vkGetDeviceQueue(logicalDevice, familyIndices.presentFamily.value(), 0, &presentQueue);
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice physicalDevice, VkSurfaceKHR vkSurface)
{
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);

	bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

	bool swapChainAdequate = false;
	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = VulkanSwapchain::querySwapChainSupport(physicalDevice, surface);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

	return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy && supportedFeatures.fillModeNonSolid;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice)
{
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions)
	{
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

QueueFamilyIndices VulkanDevice::findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	indices.graphicsFamily = getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT, queueFamilies);
	indices.computeFamily = getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT, queueFamilies);
	indices.transferFamily = getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT, queueFamilies);
	for (uint32_t i = 0; i < queueFamilyCount; i++)
	{
		VkBool32 presentSupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
		if (presentSupport)
		{
			indices.presentFamily = i;
			// Prefer a queue family that supports both graphics and presentation
			if (indices.graphicsFamily.has_value() && indices.graphicsFamily.value() == i)
			{
				break;
			}
		}
	}

	return indices;
}

uint32_t VulkanDevice::getQueueFamilyIndex(VkQueueFlags queueFlags, std::vector<VkQueueFamilyProperties> familyProperties)
{
	if ((queueFlags & VK_QUEUE_COMPUTE_BIT) == queueFlags) // checks if queueFlags has exactly the compute bit set
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(familyProperties.size()); i++)
		{
			if ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
			{
				return i;
			}
		}
	}

	// Dedicated queue for transfer
	// Try to find a queue family index that supports transfer but not graphics and compute
	if ((queueFlags & VK_QUEUE_TRANSFER_BIT) == queueFlags)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(familyProperties.size()); i++)
		{
			if ((familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && ((familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
			{
				return i;
			}
		}
	}

	// For other queue types or if no separate compute queue is present, reutrn the first one to support the requested flags
	for (uint32_t i = 0; i < static_cast<uint32_t>(familyProperties.size()); i++)
	{
		if ((familyProperties[i].queueFlags & queueFlags) == queueFlags)
		{
			return i;
		}
	}

	throw std::runtime_error("Could not find a matching queue family index");
}

VkCommandPool VulkanDevice::createCommandPool(VkDevice device, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags)
{
	VkCommandPoolCreateInfo cmdPoolInfo{};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
	cmdPoolInfo.flags = createFlags;
	VkCommandPool cmdPool;
	VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
	return cmdPool;
}

std::vector<VkCommandBuffer> VulkanDevice::createCommandBuffers(VkDevice device, VkCommandBufferLevel level, VkCommandPool pool, uint32_t count)
{
	std::vector<VkCommandBuffer> cmdBuffers(count);

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = level;
	allocInfo.commandPool = pool;
	allocInfo.commandBufferCount = count;

	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, cmdBuffers.data()));
	return cmdBuffers;
}

VkCommandBuffer VulkanDevice::createCommandBuffer(VkDevice device, VkCommandBufferLevel level, VkCommandPool pool, bool begin)
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = level;
	allocInfo.commandPool = pool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer;
	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer));

	if (begin)
	{
		VkCommandBufferBeginInfo cmdBufInfo{};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	}

	return cmdBuffer;
}

