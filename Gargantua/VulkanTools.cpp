/* This code is from SaschaWillems Vulkan/base/VulkanTools.cpp */

// added couple functions myself.

#include "VulkanTools.h"

const std::string getAssetPath()
{
	return "assets/";
}

const std::string getShaderBasePath()
{
	return "shaders/";
}

namespace vks
{
	namespace tools
	{
		bool errorModeSilent = false;

		std::string errorString(VkResult errorCode)
		{
			switch (errorCode)
			{
#define STR(r) case VK_ ##r: return #r
				STR(NOT_READY);
				STR(TIMEOUT);
				STR(EVENT_SET);
				STR(EVENT_RESET);
				STR(INCOMPLETE);
				STR(ERROR_OUT_OF_HOST_MEMORY);
				STR(ERROR_OUT_OF_DEVICE_MEMORY);
				STR(ERROR_INITIALIZATION_FAILED);
				STR(ERROR_DEVICE_LOST);
				STR(ERROR_MEMORY_MAP_FAILED);
				STR(ERROR_LAYER_NOT_PRESENT);
				STR(ERROR_EXTENSION_NOT_PRESENT);
				STR(ERROR_FEATURE_NOT_PRESENT);
				STR(ERROR_INCOMPATIBLE_DRIVER);
				STR(ERROR_TOO_MANY_OBJECTS);
				STR(ERROR_FORMAT_NOT_SUPPORTED);
				STR(ERROR_SURFACE_LOST_KHR);
				STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
				STR(SUBOPTIMAL_KHR);
				STR(ERROR_OUT_OF_DATE_KHR);
				STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
				STR(ERROR_VALIDATION_FAILED_EXT);
				STR(ERROR_INVALID_SHADER_NV);
				STR(ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
#undef STR
			default:
				return "UNKNOWN ERROR";
			}
		}

		std::string physicalDeviceTypeString(VkPhysicalDeviceType type)
		{
			switch (type)
			{
#define STR(r) case VK_PHYSICAL_DEVICE_TYPE_ ##r: return #r
				STR(OTHER);
				STR(INTEGRATED_GPU);
				STR(DISCRETE_GPU);
				STR(VIRTUAL_GPU);
				STR(CPU);
#undef STR
			default:
				return "UNKNOWN_DEVICE_TYPE";
			}
		}

		uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
		{
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
			{
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}
			throw std::runtime_error("failed to find suitable memory type!");
		}

		VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat)
		{
			// Since all depth formats may be optional, we need to find a suitable depth format to use
			// Start with the highest precision packed format
			std::vector<VkFormat> formatList = {
				VK_FORMAT_D32_SFLOAT_S8_UINT,
				VK_FORMAT_D32_SFLOAT,
				VK_FORMAT_D24_UNORM_S8_UINT,
				VK_FORMAT_D16_UNORM_S8_UINT,
				VK_FORMAT_D16_UNORM
			};

			for (auto& format : formatList)
			{
				VkFormatProperties formatProps;
				vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
				if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
				{
					*depthFormat = format;
					return true;
				}
			}

			return false;
		}

		VkBool32 getSupportedDepthStencilFormat(VkPhysicalDevice physicalDevice, VkFormat* depthStencilFormat)
		{
			std::vector<VkFormat> formatList = {
				VK_FORMAT_D32_SFLOAT_S8_UINT,
				VK_FORMAT_D24_UNORM_S8_UINT,
				VK_FORMAT_D16_UNORM_S8_UINT,
			};

			for (auto& format : formatList)
			{
				VkFormatProperties formatProps;
				vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
				if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
				{
					*depthStencilFormat = format;
					return true;
				}
			}

			return false;
		}

		VkBool32 formatIsFilterable(VkPhysicalDevice physicalDevice, VkFormat format, VkImageTiling tiling)
		{
			VkFormatProperties formatProps;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
			if (tiling == VK_IMAGE_TILING_OPTIMAL)
			{
				return formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
			}
			if (tiling == VK_IMAGE_TILING_LINEAR)
			{
				return formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
			}
			return false;
		}

		VkBool32 formatHasStencil(VkFormat format)
		{
			std::vector<VkFormat> stencilFormats = {
				VK_FORMAT_S8_UINT,
				VK_FORMAT_D16_UNORM_S8_UINT,
				VK_FORMAT_D24_UNORM_S8_UINT,
				VK_FORMAT_D32_SFLOAT_S8_UINT
			};
			return std::find(stencilFormats.begin(), stencilFormats.end(), format) != std::end(stencilFormats);
		}

		// Create an image memory barrier for changing the layout of
		// an image and put it into an active command buffer
		// See chapter 11.4 "Image Layout" for details

		void setImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkImageSubresourceRange subresourceRange,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask)
		{
			// Create an image barrier object
			VkImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
			imageMemoryBarrier.oldLayout = oldImageLayout;
			imageMemoryBarrier.newLayout = newImageLayout;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = subresourceRange;

			// Source layouts (old)
			// Source access mask controls actions that have to be finished on the old layout
			// before it will be transitioned tot he new layout
			switch (oldImageLayout)
			{
			case VK_IMAGE_LAYOUT_UNDEFINED:
				// Image layout is undefined (or does not matter)
				// Only valid as initial layout
				// No flags required, listed only for completeness
				imageMemoryBarrier.srcAccessMask = 0;
				break;

			case VK_IMAGE_LAYOUT_PREINITIALIZED:
				// Image is preinitialized
				// Only valid as initial layout for linear images, preserves memory contents
				// Make sure host writes have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
				// Image is a color attachment
				// Make sure any writes to the color buffer have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				// Image is a depth/stencil attachment
				// Make sure any writes to the depth/stencil buffer have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
				// Image is a transfer source
				// Make sure any reads from the image have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				// Image is a transfer destination
				// Make sure any writes to the image have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				// Image is read by a shader
				// Make sure any shader reads from the image have been finished
				imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
				break;

			default:
				// Other source layouts aren't handled (yet)
				break;
			}

			// Target layouts (new)
			// Destination access mask controls the dependency for the new image layout
			switch (newImageLayout)
			{
			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				// Image will be used as a transfer destination
				// Make sure any writes to the image have been finished
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
				// Image will be used as a transfer source
				// Make sure any reads from the image have been finished
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				break;

			case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
				// Image will be used as a color attachment
				// Make sure any writes to the color buffer have been finished
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				// Image layout will be used as a depth/stencil attachment
				// Make sure any writes to the depth/stencil buffer have been finished
				imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				// Image will be read in a shader (sampler, input attachment)
				// Make sure any writes tot he image have been finished
				if (imageMemoryBarrier.srcAccessMask == 0)
				{
					imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
				}
				imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				break;
			default:
				// Other destination layouts aren't handled
				break;
			}

			// Put barrier inside setup command buffer
			vkCmdPipelineBarrier(
				cmdbuffer,
				srcStageMask,
				dstStageMask,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier
			);
		}

		void setImageLayout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageAspectFlags aspectMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask)
		{
			VkImageSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = aspectMask;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1;
			subresourceRange.layerCount = 1;
			setImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
		}

		void insertImageMemoryBarrier(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkAccessFlags srcAccessMask,
			VkAccessFlags dstAccessMask,
			VkImageLayout oldImageLayout,
			VkImageLayout newImageLayout,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask,
			VkImageSubresourceRange subresourceRange)
		{
			VkImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();
			imageMemoryBarrier.srcAccessMask = srcAccessMask;
			imageMemoryBarrier.dstAccessMask = dstAccessMask;
			imageMemoryBarrier.oldLayout = oldImageLayout;
			imageMemoryBarrier.newLayout = newImageLayout;
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = subresourceRange;

			vkCmdPipelineBarrier(
				cmdbuffer,
				srcStageMask,
				dstStageMask,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier
			);
		}

		void insertBufferMemoryBarrier(
			VkCommandBuffer cmdbuffer,
			VkAccessFlags srcAccessMask,
			VkAccessFlags dstAccessMask,
			VkBuffer buffer,
			VkDeviceSize offset,
			VkDeviceSize size,
			VkPipelineStageFlags srcStageMask,
			VkPipelineStageFlags dstStageMask)
		{
			VkBufferMemoryBarrier bufferMemoryBarrier = vks::initializers::bufferMemoryBarrier();
			bufferMemoryBarrier.srcAccessMask = srcAccessMask;
			bufferMemoryBarrier.dstAccessMask = dstAccessMask;
			bufferMemoryBarrier.buffer = buffer;
			bufferMemoryBarrier.offset = offset;
			bufferMemoryBarrier.size = size;

			vkCmdPipelineBarrier(
				cmdbuffer,
				srcStageMask,
				dstAccessMask,
				0,
				0, nullptr,
				1, &bufferMemoryBarrier,
				0, nullptr
			);
		}

		void insertMemoryBarrier2(
			VkCommandBuffer cmdbuffer,
			VkAccessFlags2 srcAccessMask,
			VkAccessFlags2 dstAccessMask,
			VkPipelineStageFlags2 srcStageMask,
			VkPipelineStageFlags2 dstStageMask
		) {
			VkMemoryBarrier2 bufferMemoryBarrier{};
			bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
			bufferMemoryBarrier.srcAccessMask = srcAccessMask;
			bufferMemoryBarrier.dstAccessMask = dstAccessMask;
			bufferMemoryBarrier.srcStageMask = srcStageMask;
			bufferMemoryBarrier.dstStageMask = dstStageMask;

			VkDependencyInfo dependencyInfo{};
			dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
			dependencyInfo.memoryBarrierCount = 1;
			dependencyInfo.pMemoryBarriers = &bufferMemoryBarrier;

			vkCmdPipelineBarrier2(
				cmdbuffer,
				&dependencyInfo
			);
		}

		void exitFatal(const std::string& message, int32_t exitCode)
		{
			std::cerr << message << "\n";
		}

		void exitFatal(const std::string& message, VkResult resultCode)
		{
			exitFatal(message, (int32_t)resultCode);
		}

		VkShaderModule loadShader(const char* fileName, VkDevice device)
		{
			std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

			if (is.is_open())
			{
				size_t size = is.tellg();
				is.seekg(0, std::ios::beg);
				char* shaderCode = new char[size];
				is.read(shaderCode, size);
				is.close();

				assert(size > 0);

				VkShaderModule shaderModule;
				VkShaderModuleCreateInfo moduleCreateInfo{};
				moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				moduleCreateInfo.codeSize = size;
				moduleCreateInfo.pCode = (uint32_t*)shaderCode;

				VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule));

				delete[] shaderCode;

				return shaderModule;
			}
			else
			{
				std::cerr << "Error: Could not open shader file\"" << fileName << "\"" << "\n";
				return VK_NULL_HANDLE;
			}
		}

		std::string readFile(const char* filePath) {
			std::ifstream file(filePath);
			if (!file.is_open()) {
				throw std::runtime_error("Failed to open shader file: " + std::string(filePath));
			}
			std::string buffer;
			std::string line;
			while (std::getline(file, line)) {
				buffer += line + "\n";
			}
			if (!file.eof() && file.fail()) {
				throw std::runtime_error("Error reading shader file: " + std::string(filePath));
			}
			return buffer;
		}

		VkShaderModule loadSlangShader(VkDevice device, slang::IGlobalSession* slangGlobalSession, const char* shaderPath, const char* entryPointName)
		{

			//std::string shaderString = readFile(shaderPath);

			slang::SessionDesc sessionDesc = {};
			
			slang::TargetDesc targetDesc = {};
			targetDesc.format = SLANG_SPIRV;
			targetDesc.profile = slangGlobalSession->findProfile("spirv_1_5");
			
			sessionDesc.targets = &targetDesc;
			sessionDesc.targetCount = 1;

			/*std::array<slang::PreprocessorMacroDesc, 2> preprocessorMacroDesc =
			{
				{ "BIAS_VALUE", "1138" },
				{ "OTHER_MACRO", "float" }
			};*/

			std::array<slang::CompilerOptionEntry, 1> options =
			{
				{
					slang::CompilerOptionName::EmitSpirvDirectly,
					{slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
				}
			};

			sessionDesc.compilerOptionEntries = options.data();
			sessionDesc.compilerOptionEntryCount = options.size();

			Slang::ComPtr<slang::ISession> session;
			slangGlobalSession->createSession(sessionDesc, session.writeRef());

			// Load module
			Slang::ComPtr<slang::IModule> slangModule;
			{
				Slang::ComPtr<slang::IBlob> diagnosticsBlob;
				/*const char* moduleName = shaderPath;
				const char* modulePath = shaderPath;
				slangModule = session->loadModuleFromSourceString(
					moduleName, 
					modulePath, 
					shaderString.c_str(), 
					diagnosticsBlob.writeRef()
				);*/
				slangModule = session->loadModule(shaderPath, diagnosticsBlob.writeRef());
				diagnoseIfNeeded(diagnosticsBlob);
				if (!slangModule)
				{
					return VK_NULL_HANDLE;
				}
			}

			// Query Entry Points
			Slang::ComPtr<slang::IEntryPoint> entryPoint;
			{
				Slang::ComPtr<slang::IBlob> diagnosticsBlob;
				slangModule->findEntryPointByName(entryPointName, entryPoint.writeRef());
				if (!entryPoint)
				{
					std::cout << "Error getting entry point" << std::endl;
					return VK_NULL_HANDLE;
				}
			}

			// Compose Modules + Entry points
			std::array<slang::IComponentType*, 2> componentTypes = {
				slangModule,
				entryPoint
			};

			Slang::ComPtr<slang::IComponentType> composedProgram;
			{
				Slang::ComPtr<slang::IBlob> diagnosticsBlob;
				SlangResult result = session->createCompositeComponentType(
					componentTypes.data(),
					componentTypes.size(),
					composedProgram.writeRef(),
					diagnosticsBlob.writeRef()
				);
				diagnoseIfNeeded(diagnosticsBlob);
				if (SLANG_FAILED(result))
				{
					return VK_NULL_HANDLE;
				}
			}
			
			// Link
			Slang::ComPtr<slang::IComponentType> linkedProgram;
			{
				Slang::ComPtr<slang::IBlob> diagnosticsBlob;
				SlangResult result = composedProgram->link(
					linkedProgram.writeRef(),
					diagnosticsBlob.writeRef()
				);
				diagnoseIfNeeded(diagnosticsBlob);
				if (SLANG_FAILED(result))
				{
					return VK_NULL_HANDLE;
				}
			}
			
			// Get Target Kernel Code
			Slang::ComPtr<slang::IBlob> spirvCode;
			{
				Slang::ComPtr<slang::IBlob> diagnosticsBlob;
				SlangResult result = linkedProgram->getEntryPointCode(
					0,
					0,
					spirvCode.writeRef(),
					diagnosticsBlob.writeRef()
				);
				diagnoseIfNeeded(diagnosticsBlob);
				if (SLANG_FAILED(result))
				{
					return VK_NULL_HANDLE;
				}
			}

			VkShaderModule shaderModule;
			VkShaderModuleCreateInfo moduleCreateInfo{};
			moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			moduleCreateInfo.codeSize = spirvCode->getBufferSize();
			moduleCreateInfo.pCode = static_cast<const uint32_t*>(spirvCode->getBufferPointer());
			

			VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule));

			return shaderModule;
		}

		void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
		{
			if (diagnosticsBlob != nullptr)
			{
				std::cout << (const char*)diagnosticsBlob->getBufferPointer() << std::endl;
			}
		}

		VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool)
		{
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = commandPool;
			allocInfo.commandBufferCount = 1;
			
			VkCommandBuffer cmdBuffer;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer));

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
			
			return cmdBuffer;
		}

		void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkDevice device, VkQueue queue, VkCommandPool commandPool)
		{
			VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			VkFence fence;
			VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));


			vkQueueSubmit(queue, 1, &submitInfo, fence);
			/*vkQueueWaitIdle(queue);*/

			vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

			vkDestroyFence(device, fence, nullptr);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}


		bool fileExists(const std::string& filename)
		{
			std::ifstream f(filename.c_str());
			return !f.fail();
		}

		uint32_t alignedSize(uint32_t value, uint32_t alignment)
		{
			return (value + alignment - 1) & ~(alignment - 1);
		}

		VkDeviceSize alignedVkSize(VkDeviceSize value, VkDeviceSize alignment)
		{
			return (value + alignment - 1) & ~(alignment - 1);
		}
	}
}


