#include "BlackHoleSim.h"


void BlackHoleSim::run()
{
    initGlfwWindow();
    initVulkan();
    mainLoop();
    cleanUp();

}

void BlackHoleSim::initGlfwWindow()
{
    try {
        window = std::make_unique<Window>(width, height, windowTitle);

        window->setAppFramebufferResizeCallback([this](int width, int height)
            {
                this->resized = true;
            }
        );
        
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Window Initialization Error: " << e.what() << std::endl;
        throw;
    }
}

void BlackHoleSim::initVulkan()
{
    slang::createGlobalSession(&slangGlobalSession);

    device = std::make_unique<VulkanDevice>(window->getGlfwWindow());
    swapchain = std::make_unique<VulkanSwapchain>(device->instance, device->surface, device->logicalDevice, device->physicalDevice, window->getGlfwWindow());
    swapchain->create(width, height);

    frameCommandBuffers = VulkanDevice::createCommandBuffers(device->logicalDevice, VK_COMMAND_BUFFER_LEVEL_PRIMARY, device->graphicsCommandPool, MAX_CONCURRENT_FRAMES);
    //computeCommandBuffers = VulkanDevice::createCommandBuffers(device->logicalDevice, VK_COMMAND_BUFFER_LEVEL_PRIMARY, device->computeCommandPool, MAX_CONCURRENT_FRAMES);

    createDescriptorPool();

    initializeFrameUbo();
    //createGraphicsPipeline();
    createComputeStorageImage();
    createComputePipeline();

    //createVertexBuffer();
    //createIndexBuffer();
    //createDepthResources();

    createSyncPrimitives();

    camera = std::make_unique<Camera>();
    camera->type = Camera::CameraType::lookat;
    camera->setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    camera->setRotation(glm::vec3(0.0f));
    camera->setPerspective(60.0f, (float)width / (float)height, 1.0f, 500.0f);
}

void BlackHoleSim::mainLoop()
{
    while (window && !window->shouldClose())
    {
        window->pollEvents();
        drawFrame();
    }
}

void BlackHoleSim::cleanUp()
{
    vkDeviceWaitIdle(device->logicalDevice);

    for (size_t i = 0; i < presentCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, presentCompleteSemaphores[i], nullptr);
    for (size_t i = 0; i < renderCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, renderCompleteSemaphores[i], nullptr);
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) vkDestroyFence(device->logicalDevice, waitFences[i], nullptr);


    //depthImage.destroy();

    //indexBuffer.destroy();
    //vertexBuffer.destroy();
    computeStorageImage.destroy();
    cleanupComputePipeline();

    for (auto& buf : frameUBO) buf.destroy();


    /*vkDestroyPipeline(device->logicalDevice, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device->logicalDevice, graphicsPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device->logicalDevice, graphicsDescriptorSetLayout, nullptr);*/
    
    vkDestroyDescriptorPool(device->logicalDevice, descriptorPool, nullptr);

    swapchain.reset();
    device.reset();
    camera.reset();
    window.reset();
}

void BlackHoleSim::drawFrame()
{

    vkWaitForFences(device->logicalDevice, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);
    VK_CHECK_RESULT(vkResetFences(device->logicalDevice, 1, &waitFences[currentFrame]));

    uint32_t imageIndex{ 0 };
    VkResult result = swapchain->acquireNextImage(presentCompleteSemaphores[currentFrame], imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        windowResize();
        return;
    }
    else if ((result != VK_SUCCESS) && (result != VK_SUBOPTIMAL_KHR))
    {
        throw "Could not acquire the next swap chain image";
    }

    //updateGraphicsDescriptorSet(currentFrame, frameUBO[currentFrame].buffer);

    ShaderData shaderData{};
    shaderData.projectionMatrix = camera->matrices.perspective;
    shaderData.viewMatrix = camera->matrices.view;
    shaderData.modelMatrix = glm::mat4(1.0f);

    frameUBO[currentFrame].copyTo(&shaderData, sizeof(ShaderData));

    vkResetCommandBuffer(frameCommandBuffers[currentFrame], 0);
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    const VkCommandBuffer commandBuffer = frameCommandBuffers[currentFrame];
    //const VkCommandBuffer computeCommandBuffer = computeCommandBuffers[currentFrame];
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    // transition storage image to be writable by the shader
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        computeStorageImage.image,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT,  0, 1, 0, 1 }
    );

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);

    // 16x16 work group
    uint32_t groupCountX = (width + 15) / 16;
    uint32_t groupCountY = (height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    // barrier: wait for compute shader to finish before copying from the storage image
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        computeStorageImage.image,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    // barrier: transition swapchain image to be a copy destination
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    // copy the storage iamge to the swapchain image
    VkImageCopy imageCopyRegion{};
    imageCopyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    imageCopyRegion.srcOffset = { 0, 0, 0 };
    imageCopyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    imageCopyRegion.dstOffset = { 0, 0, 0 };
    imageCopyRegion.extent = { width, height, 1 };
    vkCmdCopyImage(
        commandBuffer,
        computeStorageImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapchain->images[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageCopyRegion
    );

    // transition swapchain image backt o be presentable
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    /*vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        depthImage.image,
        0,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 }
    );

    VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    colorAttachment.imageView = swapchain->imageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = { 0.0f, 0.0f, 0.2f, 0.0f };

    VkRenderingAttachmentInfo depthStencilAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthStencilAttachment.imageView = depthImage.imageView;
    depthStencilAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthStencilAttachment.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO_KHR };
    renderingInfo.renderArea = { 0, 0, width, height };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthStencilAttachment;
    renderingInfo.pStencilAttachment = &depthStencilAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{ 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{ 0, 0, width, height };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        graphicsPipelineLayout,
        0, 1,
        &graphicsDescriptorSets[currentFrame],
        0, nullptr
    );

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkDeviceSize offsets[1]{ 0 };

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRendering(commandBuffer);

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );*/

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    //VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pWaitDstStageMask = &waitStageMask;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.commandBufferCount = 1;
    submitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderCompleteSemaphores[imageIndex];
    submitInfo.signalSemaphoreCount = 1;

    VK_CHECK_RESULT(vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, waitFences[currentFrame]));

    result = swapchain->queuePresent(device->presentQueue, imageIndex, renderCompleteSemaphores[imageIndex]);

    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR))
    {
        windowResize();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Could not present the image to the swap chain");
    }

    currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
}

void BlackHoleSim::windowResize()
{
    int newWidth = 0, newHeight = 0;
    window->getFramebufferSize(newWidth, newHeight);
    while (newWidth == 0 || newHeight == 0)
    {
        glfwWaitEvents();
        window->getFramebufferSize(newWidth, newHeight);
        // todo: implement window->waitEvents() wait if window is minimized
    }
    vkDeviceWaitIdle(device->logicalDevice);
    this->width = newWidth;
    this->height = newHeight;

    // depthResource cleanup
    //depthImage.destroy();

    computeStorageImage.destroy();
    swapchain.reset();

    swapchain = std::make_unique<VulkanSwapchain>(device->instance, device->surface, device->logicalDevice, device->physicalDevice, window->getGlfwWindow());
    swapchain->create(this->width, this->height);
    createComputeStorageImage();
    //computeStorageImage.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    updateComputeDescriptorSets();

    // depthResource recreate
    //createDepthResources();

    camera->setPerspective(60.0f, (float)width / (float)height, 1.0f, 500.0f);
}

void BlackHoleSim::createSyncPrimitives()
{
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK_RESULT(vkCreateFence(device->logicalDevice, &fenceCI, nullptr, &waitFences[i]));
    }

    presentCompleteSemaphores.resize(MAX_CONCURRENT_FRAMES);
    for (auto& semaphore : presentCompleteSemaphores)
    {
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &semaphoreCI, nullptr, &semaphore));
    }

    renderCompleteSemaphores.resize(swapchain->images.size());
    for (auto& semaphore : renderCompleteSemaphores)
    {
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &semaphoreCI, nullptr, &semaphore));
    }
}

void BlackHoleSim::createComputePipeline()
{
    std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[0].descriptorCount = 1;
    
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[1].descriptorCount = 1;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    descriptorSetLayoutCI.pBindings = layoutBindings.data();
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &descriptorSetLayoutCI, nullptr, &computeDescriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCI{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCI.pSetLayouts = &computeDescriptorSetLayout;
    pipelineLayoutCI.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutCI, nullptr, &computePipelineLayout));

    VkShaderModule computeShader = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/raymarcher.slang", "computeMain");
    
    VkPipelineShaderStageCreateInfo shaderStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = computeShader;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo computePipelineCI{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    computePipelineCI.layout = computePipelineLayout;
    computePipelineCI.stage = shaderStage;
    VK_CHECK_RESULT(vkCreateComputePipelines(device->logicalDevice, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &computePipeline));

    vkDestroyShaderModule(device->logicalDevice, computeShader, nullptr);

    allocateDescriptorSets(device->logicalDevice, descriptorPool, computeDescriptorSetLayout, MAX_CONCURRENT_FRAMES, computeDescriptorSets);
   
    updateComputeDescriptorSets();
}

void BlackHoleSim::updateComputeDescriptorSets()
{
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        VkDescriptorImageInfo storageImageDescriptor{};
        storageImageDescriptor.imageView = computeStorageImage.imageView;
        storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo uboDescriptor{};
        uboDescriptor.buffer = frameUBO[i].buffer;
        uboDescriptor.offset = 0;
        uboDescriptor.range = sizeof(ShaderData);

        std::array<VkWriteDescriptorSet, 2> computeWriteDescriptorSets{};
        computeWriteDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[0].dstBinding = 0;
        computeWriteDescriptorSets[0].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        computeWriteDescriptorSets[0].descriptorCount = 1;
        computeWriteDescriptorSets[0].pImageInfo = &storageImageDescriptor;

        computeWriteDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[1].dstBinding = 1;
        computeWriteDescriptorSets[1].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        computeWriteDescriptorSets[1].descriptorCount = 1;
        computeWriteDescriptorSets[1].pBufferInfo = &uboDescriptor;

        vkUpdateDescriptorSets(device->logicalDevice, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
    }
}

void BlackHoleSim::cleanupComputePipeline()
{
    if (computeDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device->logicalDevice, computeDescriptorSetLayout, nullptr);
        computeDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (computePipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device->logicalDevice, computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }
    if (computePipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device->logicalDevice, computePipeline, nullptr);
        computePipeline = VK_NULL_HANDLE;
    }
}

void BlackHoleSim::createComputeStorageImage()
{
    computeStorageImage.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    computeStorageImage.imageInfo.extent.width = swapchain->extent.width;
    computeStorageImage.imageInfo.extent.height = swapchain->extent.height;
    computeStorageImage.imageInfo.extent.depth = 1;
    //computeStorageImage.imageInfo.format = swapchain->colorFormat;
    computeStorageImage.imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    computeStorageImage.imageInfo.mipLevels = 1;
    computeStorageImage.imageInfo.arrayLayers = 1;
    computeStorageImage.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    computeStorageImage.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // note the usage
    computeStorageImage.imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    computeStorageImage.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    computeStorageImage.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    //computeStorageImage.viewInfo.format = swapchain->colorFormat;
    computeStorageImage.viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;

    computeStorageImage.viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    computeStorageImage.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

//void BlackHoleSim::createGraphicsPipeline()
//{
//    VkDescriptorSetLayoutBinding uboLayoutBinding{};
//    uboLayoutBinding.binding = 0;
//    uboLayoutBinding.descriptorCount = 1;
//    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
//    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
//
//    std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
//        uboLayoutBinding
//    };
//
//    VkDescriptorSetLayoutCreateInfo layoutInfo{};
//    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
//    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
//    layoutInfo.pBindings = bindings.data();
//
//    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &layoutInfo, nullptr, &graphicsDescriptorSetLayout));
//
//    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
//    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
//    pipelineLayoutInfo.setLayoutCount = 1;
//    pipelineLayoutInfo.pSetLayouts = &graphicsDescriptorSetLayout;
//    pipelineLayoutInfo.pushConstantRangeCount = 0;
//    pipelineLayoutInfo.pPushConstantRanges = nullptr;
//
//    VK_CHECK_RESULT(vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout));
//
//    VkShaderModule vertShaderModule = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/shader.slang", "vertexMain");
//    VkShaderModule fragShaderModule = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/shader.slang", "fragmentMain");
//
//    /*VkShaderModule vertShaderModule = vks::tools::loadShader("shaders/shader.vert.spv", device->logicalDevice);
//    VkShaderModule fragShaderModule = vks::tools::loadShader("shaders/shader.frag.spv", device->logicalDevice);*/
//
//    VkPipelineShaderStageCreateInfo vertShaderStageCI{};
//    vertShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//    vertShaderStageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
//    vertShaderStageCI.module = vertShaderModule;
//    vertShaderStageCI.pName = "main";
//    VkPipelineShaderStageCreateInfo fragShaderStageCI{};
//    fragShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//    fragShaderStageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
//    fragShaderStageCI.module = fragShaderModule;
//    fragShaderStageCI.pName = "main";
//
//    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
//        vertShaderStageCI,
//        fragShaderStageCI
//    };
//
//    auto bindingDescription = Vertex::getBindingDescription();
//    auto attribDescription = Vertex::getAttributeDescriptions();
//
//    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
//    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
//    vertexInputInfo.vertexBindingDescriptionCount = 1;
//    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescription.size());
//    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
//    vertexInputInfo.pVertexAttributeDescriptions = attribDescription.data();
//
//    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
//    inputAssemblyCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
//    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
//    inputAssemblyCI.primitiveRestartEnable = VK_FALSE;
//
//    VkPipelineViewportStateCreateInfo viewportStateCI{};
//    viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
//    viewportStateCI.viewportCount = 1;
//    viewportStateCI.scissorCount = 1;
//
//    VkPipelineRasterizationStateCreateInfo rasterizerStateCI{};
//    rasterizerStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
//    rasterizerStateCI.depthClampEnable = VK_FALSE;
//    rasterizerStateCI.polygonMode = VK_POLYGON_MODE_FILL;
//    rasterizerStateCI.lineWidth = 1.0f;
//    rasterizerStateCI.cullMode = VK_CULL_MODE_NONE;
//    rasterizerStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
//    rasterizerStateCI.depthBiasEnable = VK_FALSE;
//    rasterizerStateCI.depthBiasConstantFactor = 0.0f;
//    rasterizerStateCI.depthBiasClamp = 0.0f;
//    rasterizerStateCI.depthBiasSlopeFactor = 0.0f;
//
//    VkPipelineMultisampleStateCreateInfo multisamplingStateCI{};
//    multisamplingStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
//    multisamplingStateCI.sampleShadingEnable = VK_FALSE;
//    multisamplingStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
//    multisamplingStateCI.minSampleShading = 1.0f;
//    multisamplingStateCI.pSampleMask = nullptr;
//    multisamplingStateCI.alphaToCoverageEnable = VK_FALSE;
//    multisamplingStateCI.alphaToOneEnable = VK_FALSE;
//
//    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
//    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
//    colorBlendAttachment.blendEnable = VK_FALSE;
//    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
//    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
//    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
//    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
//    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
//    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
//
//    VkPipelineColorBlendStateCreateInfo colorBlendingStateCI{};
//    colorBlendingStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
//    colorBlendingStateCI.logicOpEnable = VK_FALSE;
//    colorBlendingStateCI.logicOp = VK_LOGIC_OP_COPY; // Optional;
//    colorBlendingStateCI.attachmentCount = 1;
//    colorBlendingStateCI.pAttachments = &colorBlendAttachment;
//    colorBlendingStateCI.blendConstants[0] = 0.0f; // Optional
//    colorBlendingStateCI.blendConstants[1] = 0.0f; // Optional
//    colorBlendingStateCI.blendConstants[2] = 0.0f; // Optional
//    colorBlendingStateCI.blendConstants[3] = 0.0f; // Optional
//
//    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
//    depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
//    depthStencilStateCI.depthTestEnable = VK_TRUE;
//    depthStencilStateCI.depthWriteEnable = VK_TRUE;
//    depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS;
//
//    depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
//    depthStencilStateCI.minDepthBounds = 0.0f; // Optional
//    depthStencilStateCI.maxDepthBounds = 1.0f; // Optional
//
//    depthStencilStateCI.stencilTestEnable = VK_FALSE;
//    depthStencilStateCI.front = {}; // Optional
//    depthStencilStateCI.back = {}; // Optional
//
//    std::vector<VkDynamicState> dynamicStates = {
//        VK_DYNAMIC_STATE_VIEWPORT,
//        VK_DYNAMIC_STATE_SCISSOR
//    };
//    VkPipelineDynamicStateCreateInfo dynamicState{};
//    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
//    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
//    dynamicState.pDynamicStates = dynamicStates.data();
//
//    VkFormat depthFormat{};
//    vks::tools::getSupportedDepthFormat(device->physicalDevice, &depthFormat);
//
//    // dynamic rendering info
//    VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
//    pipelineRenderingCI.colorAttachmentCount = 1;
//    pipelineRenderingCI.pColorAttachmentFormats = &swapchain->colorFormat;
//    pipelineRenderingCI.depthAttachmentFormat = depthFormat;
//    pipelineRenderingCI.stencilAttachmentFormat = depthFormat;
//
//    VkGraphicsPipelineCreateInfo pipelineCI{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
//    pipelineCI.layout = graphicsPipelineLayout;
//    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
//    pipelineCI.pStages = shaderStages.data();
//    pipelineCI.pVertexInputState = &vertexInputInfo;
//    pipelineCI.pInputAssemblyState = &inputAssemblyCI;
//    pipelineCI.pViewportState = &viewportStateCI;
//    pipelineCI.pRasterizationState = &rasterizerStateCI;
//    pipelineCI.pMultisampleState = &multisamplingStateCI;
//    pipelineCI.pDepthStencilState = &depthStencilStateCI;
//    pipelineCI.pColorBlendState = &colorBlendingStateCI;
//    pipelineCI.pDynamicState = &dynamicState;
//    pipelineCI.pNext = &pipelineRenderingCI;
//
//    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &graphicsPipeline));
//
//    vkDestroyShaderModule(device->logicalDevice, vertShaderModule, nullptr);
//    vkDestroyShaderModule(device->logicalDevice, fragShaderModule, nullptr);
//
//    allocateDescriptorSets(device->logicalDevice, descriptorPool, graphicsDescriptorSetLayout, MAX_CONCURRENT_FRAMES, graphicsDescriptorSets);
//}

//void BlackHoleSim::updateGraphicsDescriptorSet(uint32_t currentFrameIndex, VkBuffer uboBuffer)
//{
//    VkDescriptorBufferInfo uboInfo{};
//    uboInfo.buffer = uboBuffer;
//    uboInfo.offset = 0;
//    uboInfo.range = sizeof(ShaderData);
//
//    std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
//    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
//    descriptorWrites[0].dstSet = graphicsDescriptorSets[currentFrameIndex];
//    descriptorWrites[0].dstBinding = 0;
//    descriptorWrites[0].dstArrayElement = 0;
//    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
//    descriptorWrites[0].descriptorCount = 1;
//    descriptorWrites[0].pBufferInfo = &uboInfo;
//
//    vkUpdateDescriptorSets(device->logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
//}

void BlackHoleSim::initializeFrameUbo()
{
    frameUBO.resize(MAX_CONCURRENT_FRAMES);

    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        frameUBO[i].create(
            device->logicalDevice,
            device->physicalDevice,
            sizeof(ShaderData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        frameUBO[i].map();
    }
}



void BlackHoleSim::allocateDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, uint32_t setCount, std::vector<VkDescriptorSet>& descriptorSets)
{
    descriptorSets.resize(setCount);

    std::vector<VkDescriptorSetLayout> layouts(setCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));
}

void BlackHoleSim::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_CONCURRENT_FRAMES }
    };
    
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCI.pPoolSizes = poolSizes.data();
    poolCI.maxSets = MAX_CONCURRENT_FRAMES * 2;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCI, nullptr, &descriptorPool));
}

//void BlackHoleSim::createDepthResources()
//{
//    VkFormat depthFormat{};
//    vks::tools::getSupportedDepthFormat(device->physicalDevice, &depthFormat);
//
//    //depthImage.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
//    depthImage.imageInfo.imageType = VK_IMAGE_TYPE_2D;
//    depthImage.imageInfo.extent.width = swapchain->extent.width;
//    depthImage.imageInfo.extent.height = swapchain->extent.height;
//    depthImage.imageInfo.extent.depth = 1;
//    depthImage.imageInfo.mipLevels = 1;
//    depthImage.imageInfo.arrayLayers = 1;
//    depthImage.imageInfo.format = depthFormat;
//    depthImage.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
//    depthImage.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//    depthImage.imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
//    depthImage.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
//    depthImage.imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//    depthImage.imageInfo.flags = 0;
//
//    depthImage.viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
//    //depthImage.viewInfo.image = depthImage;
//    depthImage.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
//    depthImage.viewInfo.format = depthFormat;
//
//    depthImage.viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
//    depthImage.viewInfo.subresourceRange.baseMipLevel = 0;
//    depthImage.viewInfo.subresourceRange.levelCount = 1;
//    depthImage.viewInfo.subresourceRange.baseArrayLayer = 0;
//    depthImage.viewInfo.subresourceRange.layerCount = 1;
//
//    depthImage.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
//}

//void BlackHoleSim::createVertexBuffer()
//{
//    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();
//
//    vks::Buffer staging;
//    staging.create(
//        device->logicalDevice,
//        device->physicalDevice,
//        bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//    );
//    staging.map();
//    memcpy(staging.mapped, vertices.data(), (size_t)bufferSize);
//    staging.unmap();
//
//    vertexBuffer.create(
//        device->logicalDevice,
//        device->physicalDevice,
//        bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
//        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
//    );
//
//    VkCommandBuffer cmdBuffer = vks::tools::beginSingleTimeCommands(device->logicalDevice, device->transferCommandPool);
//    VkBufferCopy copyRegion{};
//    copyRegion.size = bufferSize;
//    vkCmdCopyBuffer(cmdBuffer, staging.buffer, vertexBuffer.buffer, 1, &copyRegion);
//    vks::tools::endSingleTimeCommands(cmdBuffer, device->logicalDevice, device->transferQueue, device->transferCommandPool);
//
//    staging.destroy();
//}

//void BlackHoleSim::createIndexBuffer()
//{
//    VkDeviceSize bufferSize = sizeof(uint32_t) * indices.size();
//
//    vks::Buffer staging;
//    staging.create(
//        device->logicalDevice,
//        device->physicalDevice,
//        bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//    );
//    staging.map();
//    memcpy(staging.mapped, indices.data(), (size_t)bufferSize);
//    staging.unmap();
//
//    indexBuffer.create(
//        device->logicalDevice,
//        device->physicalDevice,
//        bufferSize,
//        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
//        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
//    );
//
//    VkCommandBuffer cmdBuffer = vks::tools::beginSingleTimeCommands(device->logicalDevice, device->transferCommandPool);
//    VkBufferCopy copyRegion{};
//    copyRegion.size = bufferSize;
//    vkCmdCopyBuffer(cmdBuffer, staging.buffer, indexBuffer.buffer, 1, &copyRegion);
//    vks::tools::endSingleTimeCommands(cmdBuffer, device->logicalDevice, device->transferQueue, device->transferCommandPool);
//
//    staging.destroy();
//}




