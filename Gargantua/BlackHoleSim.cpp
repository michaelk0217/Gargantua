/*
BIG BAD MONOLITH FILE
I know, its not a small file.
If you want to read through this, I encourage to look at the header file first, then read through the file in this order:
- initVulkan()
- mainLoop()
- drawFrame()
*/

#include "BlackHoleSim.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>

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

    createDescriptorPool();

    initializeFrameUbo();

    loadEnvironmentTextures();
    createEnvironmentSampler();

    createNoiseResources(NOISE_SIZE);
    generateNoise3D(NOISE_SIZE);

    createPostProcessResources();

    // ====== main compute pipeline ======
    createComputeStorageImage();
    createComputePipeline();
    // ===================================

    createSyncPrimitives();

    camera = std::make_unique<Camera>(
        glm::vec3(-10.0f, 0.0, 0.0), // pos
        glm::vec3(0.0f, 1.0f, 0.0f), // worldupvector: set to y-up
        0.0f, // yaw : look along x axis
        0.0f, // pitch
        3.0f, // movementSpeed
        0.1f, // turnspeed
        60.0f, // fov
        (float)width / (float)height,
        0.1f,
        500.0f
    );
    //camera->type = Camera::CameraType::firstperson;
    //camera->setPosition(glm::vec3(0.0f, 0.0f, -2.5f));
    //camera->setRotation(glm::vec3(0.0f));
    //camera->setPerspective(60.0f, (float)width / (float)height, 1.0f, 500.0f);
    //camera->setRotationSpeed(3.0f);
    //camera->setMovementSpeed(5.0f);
    //camera->flipY = false;

    uiOverlay = std::make_unique<UIOverlay>(*window, *device, *swapchain);

    initializePPImageLayouts();
}

void BlackHoleSim::mainLoop()
{

    auto lastTime = std::chrono::high_resolution_clock::now();
    frame_history.resize(90, 0); // keeps track of 90 recent frame rates

    blackHoleMass = 0.4f;
    blackHoleSpin = 0.6f;
    maxSteps = 100;
    stepSize = 0.2f;
    backgroundType = 0;
    geodesicType = 0;
    diskEnable = false;

    

    while (window && !window->shouldClose())
    {
        // delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;
        totalElapsedTime += deltaTime;

        UIOverlay::update_frame_history(frame_history, 1.0f / deltaTime);

        window->pollEvents();
        processInput(deltaTime);

        /*glm::vec3 forwardDir;
        forwardDir.x = -cos(glm::radians(camera->rotation.x)) * sin(glm::radians(camera->rotation.y));
        forwardDir.y = sin(glm::radians(camera->rotation.x));
        forwardDir.z = cos(glm::radians(camera->rotation.x)) * cos(glm::radians(camera->rotation.y));
        forwardDir = glm::normalize(forwardDir);*/

        /*glm::mat4 invView = glm::inverse(camera->matrices.view);
        glm::vec3 forwardDir = -glm::normalize(glm::vec3(invView[2]));*/

        glm::vec3 forwardDir = camera->getCameraDirection();

        UIPacket uiPacket{
            deltaTime,
            frame_history,
            forwardDir,
            blackHoleMass,
            blackHoleSpin,
            maxSteps,
            stepSize,
            backgroundType,
            geodesicType,
            diskEnable
        };
        uiOverlay->newFrame();
        uiOverlay->buildUI(uiPacket);



        drawFrame();
    }
}

void BlackHoleSim::cleanUp()
{
    vkDeviceWaitIdle(device->logicalDevice);

    uiOverlay.reset();

    for (size_t i = 0; i < presentCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, presentCompleteSemaphores[i], nullptr);
    for (size_t i = 0; i < renderCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, renderCompleteSemaphores[i], nullptr);
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) vkDestroyFence(device->logicalDevice, waitFences[i], nullptr);


    //depthImage.destroy();
    //indexBuffer.destroy();
    //vertexBuffer.destroy();

    cleanupPostProcessResources();
    destroyNoiseResources();
    spheremapTexture.destroy();
    vkDestroySampler(device->logicalDevice, environmentSampler, nullptr);
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
    shaderData.projectionMatrix = camera->getProjectionMatrix();
    shaderData.viewMatrix = camera->calculateViewMatrix();
    shaderData.modelMatrix = glm::mat4(1.0f);
    shaderData.inverseProjectionMatrix = glm::inverse(camera->getProjectionMatrix());
    shaderData.inverseViewMatrix = glm::inverse(camera->calculateViewMatrix());
    shaderData.cameraPosition = camera->getCameraPosition();
    shaderData.blackHoleMass = blackHoleMass;
    shaderData.blackHoleSpin = blackHoleSpin;
    shaderData.maxSteps = maxSteps;
    shaderData.stepSize = stepSize;
    shaderData.time = totalElapsedTime;
    shaderData.backgroundType = backgroundType;
    shaderData.geodesicType = geodesicType;
    shaderData.diskEnable = diskEnable ? 1 : 0;
    frameUBO[currentFrame].copyTo(&shaderData, sizeof(ShaderData));

   


    vkResetCommandBuffer(frameCommandBuffers[currentFrame], 0);
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    const VkCommandBuffer commandBuffer = frameCommandBuffers[currentFrame];
    //const VkCommandBuffer computeCommandBuffer = computeCommandBuffers[currentFrame];
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);

    // 16x16 work group
    uint32_t groupCountX = (width + 15) / 16;
    uint32_t groupCountY = (height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        postProcess.hdrColorBuffer.image,
        VK_ACCESS_SHADER_WRITE_BIT,             // From raymarcher write
        VK_ACCESS_SHADER_READ_BIT,              // For bright extract read
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, // Optimize for reading
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    performBloomPass(commandBuffer, currentFrame);

    performFinalComposite(commandBuffer, currentFrame, imageIndex);

    ppImageLayoutTransitionNextFrame(commandBuffer);

    //// barrier: wait for compute shader to finish before copying from the storage image
    //vks::tools::insertImageMemoryBarrier(
    //    commandBuffer,
    //    computeStorageImage.image,
    //    VK_ACCESS_SHADER_WRITE_BIT,
    //    VK_ACCESS_TRANSFER_READ_BIT,
    //    VK_IMAGE_LAYOUT_GENERAL,
    //    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    //    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //    VK_PIPELINE_STAGE_TRANSFER_BIT,
    //    VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    //);

    //// barrier: transition swapchain image to be a copy destination
    //vks::tools::insertImageMemoryBarrier(
    //    commandBuffer,
    //    swapchain->images[imageIndex],
    //    0,
    //    VK_ACCESS_TRANSFER_WRITE_BIT,
    //    VK_IMAGE_LAYOUT_UNDEFINED,
    //    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    //    VK_PIPELINE_STAGE_TRANSFER_BIT,
    //    VK_PIPELINE_STAGE_TRANSFER_BIT,
    //    VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    //);

    //// copy the storage iamge to the swapchain image
    //VkImageCopy imageCopyRegion{};
    //imageCopyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    //imageCopyRegion.srcOffset = { 0, 0, 0 };
    //imageCopyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    //imageCopyRegion.dstOffset = { 0, 0, 0 };
    //imageCopyRegion.extent = { width, height, 1 };
    //vkCmdCopyImage(
    //    commandBuffer,
    //    computeStorageImage.image,
    //    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    //    swapchain->images[imageIndex],
    //    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    //    1,
    //    &imageCopyRegion
    //);

    // transition swapchian image to be color attachment for ui overlay
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );
    
    uiOverlay->render(commandBuffer, imageIndex, width, height);

    // transition swapchain image back to be presentable
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    // transition hdr to general for next frame
    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        postProcess.hdrColorBuffer.image,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

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
    updateComputeDescriptorSets();
    
    postProcessWindowResizeRecreate();
    // depthResource recreate
    //createDepthResources();

    //camera->setPerspective(60.0f, (float)width / (float)height, 1.0f, 500.0f);
    camera->setAspectRatio((float)width / (float)height);
}

void BlackHoleSim::processInput(float deltaTime)
{
    if (window->isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE))
    {
        camera->processKeyboard(window->getKeys(), deltaTime);
        glfwSetInputMode(window->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        camera->processMouseMovement(window->getXChange(), window->getYChange(), true);
    }
    else
    {
        glfwSetInputMode(window->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    
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
    std::array<VkDescriptorSetLayoutBinding, 6> layoutBindings{};
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[0].descriptorCount = 1;
    
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[1].descriptorCount = 1;

    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[2].descriptorCount = 1;
     
    layoutBindings[3].binding = 3;
    layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[3].descriptorCount = 1;

    layoutBindings[4].binding = 4;
    layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[4].descriptorCount = 1;

    layoutBindings[5].binding = 5;
    layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings[5].descriptorCount = 1;


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
        //storageImageDescriptor.imageView = computeStorageImage.imageView;
        storageImageDescriptor.imageView = postProcess.hdrColorBuffer.imageView;
        storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo uboDescriptor{};
        uboDescriptor.buffer = frameUBO[i].buffer;
        uboDescriptor.offset = 0;
        uboDescriptor.range = sizeof(ShaderData);

        VkDescriptorImageInfo spheremapDescriptor{};
        spheremapDescriptor.imageView = spheremapTexture.imageView;
        spheremapDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        
        VkDescriptorImageInfo environmentSamplerDescriptor{};
        environmentSamplerDescriptor.sampler = environmentSampler;

        VkDescriptorImageInfo noise3dImageInfo{};
        noise3dImageInfo.imageView = noiseTexture3D.imageView;
        noise3dImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo noise3dSamplerDescriptor{};
        noise3dSamplerDescriptor.sampler = noiseSampler;

        std::array<VkWriteDescriptorSet, 6> computeWriteDescriptorSets{};
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

        computeWriteDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[2].dstBinding = 2;
        computeWriteDescriptorSets[2].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        computeWriteDescriptorSets[2].descriptorCount = 1;
        computeWriteDescriptorSets[2].pImageInfo = &spheremapDescriptor;

        computeWriteDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[3].dstBinding = 3;
        computeWriteDescriptorSets[3].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        computeWriteDescriptorSets[3].descriptorCount = 1;
        computeWriteDescriptorSets[3].pImageInfo = &environmentSamplerDescriptor;

        computeWriteDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[4].dstBinding = 4;
        computeWriteDescriptorSets[4].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        computeWriteDescriptorSets[4].descriptorCount = 1;
        computeWriteDescriptorSets[4].pImageInfo = &noise3dImageInfo;

        computeWriteDescriptorSets[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        computeWriteDescriptorSets[5].dstBinding = 5;
        computeWriteDescriptorSets[5].dstSet = computeDescriptorSets[i];
        computeWriteDescriptorSets[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        computeWriteDescriptorSets[5].descriptorCount = 1;
        computeWriteDescriptorSets[5].pImageInfo = &noise3dSamplerDescriptor;

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
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES * 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_CONCURRENT_FRAMES  * 20},
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_CONCURRENT_FRAMES * 20 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, MAX_CONCURRENT_FRAMES * 10 }
    };
    
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCI.pPoolSizes = poolSizes.data();
    poolCI.maxSets = MAX_CONCURRENT_FRAMES * 20;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCI, nullptr, &descriptorPool));
}

void BlackHoleSim::loadEnvironmentTextures()
{
    int texWidth, texHeight, texChannels;
    float* pixels = stbi_loadf("assets/HDR_rich_multi_nebulae_2.hdr", &texWidth, &texHeight, &texChannels, 4);

    if (!pixels)
    {
        std::cerr << "Failed to load HDR Texture!" << std::endl;
        createDummyHDRTexture(spheremapTexture);
        return;
    }

    VkDeviceSize imageSize = texWidth * texHeight * 4 * sizeof(float);

    vks::Buffer stagingBuffer;
    stagingBuffer.create(
        device->logicalDevice,
        device->physicalDevice,
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    stagingBuffer.map();
    memcpy(stagingBuffer.mapped, pixels, imageSize);
    stagingBuffer.unmap();

    spheremapTexture.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    spheremapTexture.imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    spheremapTexture.imageInfo.extent = { (uint32_t)texWidth, (uint32_t)texHeight, 1 };
    spheremapTexture.imageInfo.mipLevels = 1;
    spheremapTexture.imageInfo.arrayLayers = 1;
    spheremapTexture.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    spheremapTexture.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    spheremapTexture.imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    spheremapTexture.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    spheremapTexture.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    spheremapTexture.viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    spheremapTexture.viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    spheremapTexture.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBuffer copyCmd = vks::tools::beginSingleTimeCommands(device->logicalDevice, device->graphicsCommandPool);
    
    vks::tools::insertImageMemoryBarrier(
        copyCmd,
        spheremapTexture.image,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { (uint32_t)texWidth, (uint32_t)texHeight, 1 };

    vkCmdCopyBufferToImage(
        copyCmd,
        stagingBuffer.buffer,
        spheremapTexture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    vks::tools::insertImageMemoryBarrier(
        copyCmd,
        spheremapTexture.image,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::endSingleTimeCommands(copyCmd, device->logicalDevice, device->graphicsQueue, device->graphicsCommandPool);

    stagingBuffer.destroy();
    stbi_image_free(pixels);

    std::cout << "Loaded HDR TEXTURE" << std::endl;
}

void BlackHoleSim::createDummyHDRTexture(vks::Image& texture)
{
    float pixel[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; // Black in RGBA float

    vks::Buffer stagingBuffer;
    stagingBuffer.create(
        device->logicalDevice, 
        device->physicalDevice,
        sizeof(pixel), 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    stagingBuffer.map();
    memcpy(stagingBuffer.mapped, pixel, sizeof(pixel));
    stagingBuffer.unmap();

    texture.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    texture.imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    texture.imageInfo.extent = { 1, 1, 1 };
    texture.imageInfo.mipLevels = 1;
    texture.imageInfo.arrayLayers = 1;
    texture.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    texture.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    texture.imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    texture.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    texture.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    texture.viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    texture.viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    texture.createImage(device->logicalDevice, device->physicalDevice,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandBuffer copyCmd = vks::tools::beginSingleTimeCommands(device->logicalDevice, device->graphicsCommandPool);

    vks::tools::insertImageMemoryBarrier(
        copyCmd,
        texture.image,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { 1, 1, 1 };

    vkCmdCopyBufferToImage(
        copyCmd,
        stagingBuffer.buffer,
        texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    vks::tools::insertImageMemoryBarrier(
        copyCmd,
        texture.image,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::endSingleTimeCommands(copyCmd, device->logicalDevice, device->graphicsQueue, device->graphicsCommandPool);

    stagingBuffer.destroy();

    std::cout << "Loaded DUMMY HDR TEXTURE" << std::endl;
}

void BlackHoleSim::createEnvironmentSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;

    VK_CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerInfo, nullptr, &environmentSampler));
}

void BlackHoleSim::createNoiseResources(int size, VkFormat format)
{

    // Create image resources
    noiseTexture3D.imageInfo.imageType = VK_IMAGE_TYPE_3D;
    noiseTexture3D.imageInfo.extent = { (uint32_t)size, (uint32_t)size, (uint32_t)size };
    noiseTexture3D.imageInfo.mipLevels = 1;
    noiseTexture3D.imageInfo.arrayLayers = 1;
    noiseTexture3D.imageInfo.format = format;
    noiseTexture3D.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    noiseTexture3D.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    noiseTexture3D.imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    noiseTexture3D.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    noiseTexture3D.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    noiseTexture3D.viewInfo.format = format;
    noiseTexture3D.viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    noiseTexture3D.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCI.anisotropyEnable = VK_FALSE;
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerCI.unnormalizedCoordinates = VK_FALSE;
    samplerCI.compareEnable = VK_FALSE;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = 0.0f;

    VK_CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerCI, nullptr, &noiseSampler));

    // create noise param ubo
    noiseUBO.create(
        device->logicalDevice,
        device->physicalDevice,
        sizeof(NoiseUboParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    noiseUBO.map();


    std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;


    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    descriptorSetLayoutCI.pBindings = layoutBindings.data();
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &descriptorSetLayoutCI, nullptr, &noiseDescriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCI{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCI.pSetLayouts = &noiseDescriptorSetLayout;
    pipelineLayoutCI.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutCI, nullptr, &noiseComputePipelineLayout));

    VkShaderModule computeShader = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/noise3d.slang", "main");

    VkPipelineShaderStageCreateInfo shaderStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = computeShader;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo computePipelineCI{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    computePipelineCI.layout = noiseComputePipelineLayout;
    computePipelineCI.stage = shaderStage;
    VK_CHECK_RESULT(vkCreateComputePipelines(device->logicalDevice, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &noiseComputePipeline));

    vkDestroyShaderModule(device->logicalDevice, computeShader, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &noiseDescriptorSetLayout;

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &allocInfo, &noiseComputeDescriptorSet));

    VkDescriptorImageInfo storageImageDescriptor{};
    storageImageDescriptor.imageView = noiseTexture3D.imageView;
    storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo uboDescriptor{};
    uboDescriptor.buffer = noiseUBO.buffer;
    uboDescriptor.offset = 0;
    uboDescriptor.range = sizeof(NoiseUboParams);

    std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};
    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[0].dstBinding = 0;
    writeDescriptorSets[0].dstSet = noiseComputeDescriptorSet;
    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pImageInfo = &storageImageDescriptor;

    writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[1].dstBinding = 1;
    writeDescriptorSets[1].dstSet = noiseComputeDescriptorSet;
    writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSets[1].descriptorCount = 1;
    writeDescriptorSets[1].pBufferInfo = &uboDescriptor;

    vkUpdateDescriptorSets(device->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void BlackHoleSim::destroyNoiseResources()
{
    noiseTexture3D.destroy();
    vkDestroySampler(device->logicalDevice, noiseSampler, nullptr);
    noiseUBO.destroy();
    vkDestroyPipeline(device->logicalDevice, noiseComputePipeline, nullptr);
    vkDestroyPipelineLayout(device->logicalDevice, noiseComputePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device->logicalDevice, noiseDescriptorSetLayout, nullptr);
}

void BlackHoleSim::generateNoise3D(int size)
{
    NoiseUboParams params{};
    params.size = size;
    params.scale = 4.0f;        
    params.boost = 1.5f;        
    params.octaves = 4;         
    params.tileSize = size;     
    params.lacunarity = 2.0f;
    params.gain = 0.5f;
    params.time = 0.0f;

    noiseUBO.copyTo(&params, sizeof(NoiseUboParams));

    VkCommandBuffer cmd = vks::tools::beginSingleTimeCommands(device->logicalDevice, device->graphicsCommandPool);

    vks::tools::insertImageMemoryBarrier(
        cmd,
        noiseTexture3D.image,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, noiseComputePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, noiseComputePipelineLayout, 0, 1, &noiseComputeDescriptorSet, 0, nullptr);
    
    uint32_t groupSize = 8;
    uint32_t gx = (size + groupSize - 1) / groupSize;
    uint32_t gy = gx;
    uint32_t gz = gx;
    vkCmdDispatch(cmd, gx, gy, gz);

    vks::tools::insertImageMemoryBarrier(
        cmd,
        noiseTexture3D.image,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::endSingleTimeCommands(cmd, device->logicalDevice, device->graphicsQueue, device->graphicsCommandPool);
}

void BlackHoleSim::createPostProcessResources()
{
    createHDRColorBuffer();
    createCompositeOutputImage();
    createBloomMipChain();

    createPostProcessSamplers();

    //createPostProcessUBO();

    createPostProcessDescriptorLayouts();

    createPostProcessPipelines();

    allocatePostProcessDescriptorSets();
    updatePostProcessDescriptorSets();

}

void BlackHoleSim::createHDRColorBuffer()
{
    postProcess.hdrColorBuffer.imageInfo = {};
    postProcess.hdrColorBuffer.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    postProcess.hdrColorBuffer.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    postProcess.hdrColorBuffer.imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    postProcess.hdrColorBuffer.imageInfo.extent = { width, height, 1 };
    postProcess.hdrColorBuffer.imageInfo.mipLevels = 1;
    postProcess.hdrColorBuffer.imageInfo.arrayLayers = 1;
    postProcess.hdrColorBuffer.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    postProcess.hdrColorBuffer.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    postProcess.hdrColorBuffer.imageInfo.usage =
        VK_IMAGE_USAGE_STORAGE_BIT |     // Write from compute
        VK_IMAGE_USAGE_SAMPLED_BIT |     // Read in post-process
        VK_IMAGE_USAGE_TRANSFER_DST_BIT; // Clear operations
    postProcess.hdrColorBuffer.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    postProcess.hdrColorBuffer.viewInfo = {};
    postProcess.hdrColorBuffer.viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    postProcess.hdrColorBuffer.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    postProcess.hdrColorBuffer.viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    postProcess.hdrColorBuffer.viewInfo.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };

    postProcess.hdrColorBuffer.createImage(
        device->logicalDevice,
        device->physicalDevice,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    postProcess.brightPassBuffer.imageInfo = postProcess.hdrColorBuffer.imageInfo; // Copy settings
    postProcess.brightPassBuffer.viewInfo = postProcess.hdrColorBuffer.viewInfo;
    postProcess.brightPassBuffer.createImage(
        device->logicalDevice,
        device->physicalDevice,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void BlackHoleSim::createCompositeOutputImage()
{
    postProcess.compositeOutputImage.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    postProcess.compositeOutputImage.imageInfo.extent = { width, height, 1 };
    postProcess.compositeOutputImage.imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    postProcess.compositeOutputImage.imageInfo.mipLevels = 1;
    postProcess.compositeOutputImage.imageInfo.arrayLayers = 1;
    postProcess.compositeOutputImage.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    postProcess.compositeOutputImage.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    // Add STORAGE_BIT and TRANSFER_SRC_BIT for blitting
    postProcess.compositeOutputImage.imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    postProcess.compositeOutputImage.viewInfo = {};
    postProcess.compositeOutputImage.viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    postProcess.compositeOutputImage.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    postProcess.compositeOutputImage.viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    postProcess.compositeOutputImage.viewInfo.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };

    postProcess.compositeOutputImage.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void BlackHoleSim::createBloomMipChain()
{
    postProcess.bloomMipChain.resize(PostProcessResources::BLOOM_MIP_LEVELS);

    for (int i = 0; i < PostProcessResources::BLOOM_MIP_LEVELS; i++) {
        uint32_t mipWidth = std::max(1u, width >> i);
        uint32_t mipHeight = std::max(1u, height >> i);

        auto& mipImage = postProcess.bloomMipChain[i];

        mipImage.imageInfo = {};
        mipImage.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        mipImage.imageInfo.imageType = VK_IMAGE_TYPE_2D;
        mipImage.imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        mipImage.imageInfo.extent = { mipWidth, mipHeight, 1 };
        mipImage.imageInfo.mipLevels = 1;
        mipImage.imageInfo.arrayLayers = 1;
        mipImage.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        mipImage.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        mipImage.imageInfo.usage =
            VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
        mipImage.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        mipImage.viewInfo = {};
        mipImage.viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        mipImage.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        mipImage.viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        mipImage.viewInfo.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        };

        mipImage.createImage(
            device->logicalDevice,
            device->physicalDevice,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    postProcess.tempBlurBuffer.imageInfo = {};
    postProcess.tempBlurBuffer.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    postProcess.tempBlurBuffer.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    postProcess.tempBlurBuffer.imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    postProcess.tempBlurBuffer.imageInfo.extent = { width, height, 1 };
    postProcess.tempBlurBuffer.imageInfo.mipLevels = 1;
    postProcess.tempBlurBuffer.imageInfo.arrayLayers = 1;
    postProcess.tempBlurBuffer.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    postProcess.tempBlurBuffer.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    postProcess.tempBlurBuffer.imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    postProcess.tempBlurBuffer.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    postProcess.tempBlurBuffer.viewInfo = {};
    postProcess.tempBlurBuffer.viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    postProcess.tempBlurBuffer.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    postProcess.tempBlurBuffer.viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    postProcess.tempBlurBuffer.viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    postProcess.tempBlurBuffer.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void BlackHoleSim::createPostProcessSamplers()
{
    // Linear sampler for smooth filtering
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    VK_CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerInfo, nullptr,
        &postProcess.linearSampler));

    // Nearest sampler for pixel-perfect sampling
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    VK_CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerInfo, nullptr,
        &postProcess.nearestSampler));
}

void BlackHoleSim::createPostProcessDescriptorLayouts()
{
    {
        // ----- Bright extract layout -----
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        // input texture (hdr)
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // output texture (bright)
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // sampler
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // UBO
        /*bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;*/

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &layoutInfo, nullptr, &postProcess.brightExtractLayout));
    }

    {
        // ----- Blur Layout -----
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        // input texture
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // output texture
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // sampler
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // UBO
        /*bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;*/

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &layoutInfo, nullptr, &postProcess.blurLayout));
    }

    {
        // ----- Composite layout -----
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        // HDR input
        bindings[0].binding = 0; 
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // Bloom input
        bindings[1].binding = 1; 
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[1].descriptorCount = PostProcessResources::BLOOM_MIP_LEVELS;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // Output (swapchain)
        bindings[2].binding = 2; 
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // Sampler
        bindings[3].binding = 3; 
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // UBO
        /*bindings[4].binding = 4; 
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;*/

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &layoutInfo, nullptr, &postProcess.compositeLayout));
    }
}

void BlackHoleSim::createPostProcessPipelines()
{
    createPostProcessPipelineLayouts();

    // bright extract pieline
    {
        VkShaderModule shaderModule = vks::tools::loadSlangShader(
            device->logicalDevice,
            slangGlobalSession,
            "shaders/bloom_extract.slang",
            "main"
        );

        VkPipelineShaderStageCreateInfo shaderStage{};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStage.module = shaderModule;
        shaderStage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = postProcess.brightExtractPipelineLayout;
        pipelineInfo.stage = shaderStage;

        VK_CHECK_RESULT(vkCreateComputePipelines(
            device->logicalDevice,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &postProcess.brightExtractPipeline
        ));

        vkDestroyShaderModule(device->logicalDevice, shaderModule, nullptr);
    }

    // Gaussian blur horizontal + vertical pipeline
    {
        VkShaderModule shaderModule = vks::tools::loadSlangShader(
            device->logicalDevice,
            slangGlobalSession,
            "shaders/gaussian_blur.slang",
            "main"
        );

        VkPipelineShaderStageCreateInfo shaderStage{};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStage.module = shaderModule;
        shaderStage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = postProcess.blurPipelineLayout;
        pipelineInfo.stage = shaderStage;

        VK_CHECK_RESULT(vkCreateComputePipelines(
            device->logicalDevice,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &postProcess.blurHorizontalPipeline
        ));

        // Vertical uses same shader with different UBO params
        VK_CHECK_RESULT(vkCreateComputePipelines(
            device->logicalDevice,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &postProcess.blurVerticalPipeline
        ));

        vkDestroyShaderModule(device->logicalDevice, shaderModule, nullptr);
    }

    // composite pipeline
    {
        VkShaderModule shaderModule = vks::tools::loadSlangShader(
            device->logicalDevice,
            slangGlobalSession,
            "shaders/post_composite.slang",
            "main"
        );

        VkPipelineShaderStageCreateInfo shaderStage{};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStage.module = shaderModule;
        shaderStage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = postProcess.compositePipelineLayout;
        pipelineInfo.stage = shaderStage;

        VK_CHECK_RESULT(vkCreateComputePipelines(
            device->logicalDevice,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &postProcess.compositePipeline
        ));

        vkDestroyShaderModule(device->logicalDevice, shaderModule, nullptr);
    }
}

void BlackHoleSim::createPostProcessPipelineLayouts()
{
    // Bright extract pipeline layout
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &postProcess.brightExtractLayout;
        
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(BloomExtractParams);
        push.offset = 0;

        layoutInfo.pPushConstantRanges = &push;
        layoutInfo.pushConstantRangeCount = 1;

        VK_CHECK_RESULT(vkCreatePipelineLayout(
            device->logicalDevice,
            &layoutInfo,
            nullptr,
            &postProcess.brightExtractPipelineLayout
        ));
    }

    // blur pipeline layout
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &postProcess.blurLayout;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(GaussianBlurParams);
        push.offset = 0;

        layoutInfo.pPushConstantRanges = &push;
        layoutInfo.pushConstantRangeCount = 1;

        VK_CHECK_RESULT(vkCreatePipelineLayout(
            device->logicalDevice,
            &layoutInfo,
            nullptr,
            &postProcess.blurPipelineLayout
        ));
    }

    // composite pipeline layout
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &postProcess.compositeLayout;

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(CompositeParams);
        push.offset = 0;

        layoutInfo.pPushConstantRanges = &push;
        layoutInfo.pushConstantRangeCount = 1;

        VK_CHECK_RESULT(vkCreatePipelineLayout(
            device->logicalDevice,
            &layoutInfo,
            nullptr,
            &postProcess.compositePipelineLayout
        ));
    }
}

void BlackHoleSim::allocatePostProcessDescriptorSets()
{
    // Allocate bright extract sets
    postProcess.brightExtractSets.resize(MAX_CONCURRENT_FRAMES);
    allocateDescriptorSets(
        device->logicalDevice,
        descriptorPool,
        postProcess.brightExtractLayout,
        MAX_CONCURRENT_FRAMES,
        postProcess.brightExtractSets
    );

    // Allocate blur sets (per frame, per mip level)
    postProcess.blurSets.resize(MAX_CONCURRENT_FRAMES);
    for (uint32_t frame = 0; frame < MAX_CONCURRENT_FRAMES; frame++) {
        postProcess.blurSets[frame].resize(PostProcessResources::BLOOM_MIP_LEVELS * 2); // x2 for H and V
        allocateDescriptorSets(
            device->logicalDevice,
            descriptorPool,
            postProcess.blurLayout,
            PostProcessResources::BLOOM_MIP_LEVELS * 2,
            postProcess.blurSets[frame]
        );
    }

    // Allocate composite sets
    postProcess.compositeSets.resize(MAX_CONCURRENT_FRAMES);
    allocateDescriptorSets(
        device->logicalDevice,
        descriptorPool,
        postProcess.compositeLayout,
        MAX_CONCURRENT_FRAMES,
        postProcess.compositeSets
    );
}

void BlackHoleSim::updatePostProcessDescriptorSets()
{
    for (uint32_t frame = 0; frame < MAX_CONCURRENT_FRAMES; frame++) {
        // Update bright extract descriptor set
        {
            VkDescriptorImageInfo inputInfo{};
            inputInfo.imageView = postProcess.hdrColorBuffer.imageView;
            inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo outputInfo{};
            outputInfo.imageView = postProcess.brightPassBuffer.imageView;
            outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo samplerInfo{};
            samplerInfo.sampler = postProcess.linearSampler;

            /*VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = postProcess.postProcessUBO[frame].buffer;
            uboInfo.offset = 0;
            uboInfo.range = sizeof(PostProcessParams);*/

            std::array<VkWriteDescriptorSet, 3> writes{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = postProcess.brightExtractSets[frame];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[0].descriptorCount = 1;
            writes[0].pImageInfo = &inputInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = postProcess.brightExtractSets[frame];
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo = &outputInfo;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = postProcess.brightExtractSets[frame];
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[2].descriptorCount = 1;
            writes[2].pImageInfo = &samplerInfo;

            /*writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = postProcess.brightExtractSets[frame];
            writes[3].dstBinding = 3;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[3].descriptorCount = 1;
            writes[3].pBufferInfo = &uboInfo;*/

            vkUpdateDescriptorSets(
                device->logicalDevice,
                static_cast<uint32_t>(writes.size()),
                writes.data(),
                0,
                nullptr
            );
        }

        // Update blur descriptor sets for each mip level
        for (int mip = 0; mip < PostProcessResources::BLOOM_MIP_LEVELS; mip++) {
            // Horizontal blur: read from previous stage, write to mip
            {
                int setIndex = mip * 2; // Even indices for horizontal

                VkDescriptorImageInfo inputInfo{};
                if (mip == 0) {
                    inputInfo.imageView = postProcess.brightPassBuffer.imageView;
                }
                else {
                    inputInfo.imageView = postProcess.bloomMipChain[mip - 1].imageView;
                }
                inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkDescriptorImageInfo outputInfo{};
                outputInfo.imageView = postProcess.tempBlurBuffer.imageView;
                outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo samplerInfo{};
                samplerInfo.sampler = postProcess.linearSampler;

                /*VkDescriptorBufferInfo uboInfo{};
                uboInfo.buffer = postProcess.postProcessUBO[frame].buffer;
                uboInfo.offset = 0;
                uboInfo.range = sizeof(PostProcessParams);*/

                std::array<VkWriteDescriptorSet, 3> writes{};

                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstSet = postProcess.blurSets[frame][setIndex];
                writes[0].dstBinding = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                writes[0].descriptorCount = 1;
                writes[0].pImageInfo = &inputInfo;

                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstSet = postProcess.blurSets[frame][setIndex];
                writes[1].dstBinding = 1;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[1].descriptorCount = 1;
                writes[1].pImageInfo = &outputInfo;

                writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[2].dstSet = postProcess.blurSets[frame][setIndex];
                writes[2].dstBinding = 2;
                writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                writes[2].descriptorCount = 1;
                writes[2].pImageInfo = &samplerInfo;

                /*writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[3].dstSet = postProcess.blurSets[frame][setIndex];
                writes[3].dstBinding = 3;
                writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[3].descriptorCount = 1;
                writes[3].pBufferInfo = &uboInfo;*/

                vkUpdateDescriptorSets(device->logicalDevice, 3, writes.data(), 0, nullptr);
            }

            // Vertical blur: read from horizontal output, write back to same mip
            {
                int setIndex = mip * 2 + 1; // Odd indices for vertical
                
                VkDescriptorImageInfo inputInfo{};
                inputInfo.imageView = postProcess.tempBlurBuffer.imageView;
                inputInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkDescriptorImageInfo outputInfo{};
                outputInfo.imageView = postProcess.bloomMipChain[mip].imageView;
                outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo samplerInfo{};
                samplerInfo.sampler = postProcess.linearSampler;

                /*VkDescriptorBufferInfo uboInfo{};
                uboInfo.buffer = postProcess.postProcessUBO[frame].buffer;
                uboInfo.offset = 0;
                uboInfo.range = sizeof(PostProcessParams);*/

                std::array<VkWriteDescriptorSet, 3> writes{};

                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstSet = postProcess.blurSets[frame][setIndex];
                writes[0].dstBinding = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                writes[0].descriptorCount = 1;
                writes[0].pImageInfo = &inputInfo;

                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstSet = postProcess.blurSets[frame][setIndex];
                writes[1].dstBinding = 1;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[1].descriptorCount = 1;
                writes[1].pImageInfo = &outputInfo;

                writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[2].dstSet = postProcess.blurSets[frame][setIndex];
                writes[2].dstBinding = 2;
                writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                writes[2].descriptorCount = 1;
                writes[2].pImageInfo = &samplerInfo;

                /*writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[3].dstSet = postProcess.blurSets[frame][setIndex];
                writes[3].dstBinding = 3;
                writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writes[3].descriptorCount = 1;
                writes[3].pBufferInfo = &uboInfo;*/

                vkUpdateDescriptorSets(device->logicalDevice, 3, writes.data(), 0, nullptr);
            }
        }

        // 3. Update composite descriptor set
        {
            VkDescriptorImageInfo hdrInfo{};
            hdrInfo.imageView = postProcess.hdrColorBuffer.imageView;
            hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::array<VkDescriptorImageInfo, PostProcessResources::BLOOM_MIP_LEVELS> bloomInfos;
            for (int i = 0; i < PostProcessResources::BLOOM_MIP_LEVELS; i++) {
                bloomInfos[i].imageView = postProcess.bloomMipChain[i].imageView;
                bloomInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }

            // Note: Output will be set dynamically to current swapchain image
            VkDescriptorImageInfo outputInfo{};
            //outputInfo.imageView = swapchain->imageViews[0]; // Placeholder, updated per frame
            outputInfo.imageView = postProcess.compositeOutputImage.imageView;
            outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo samplerInfo{};
            samplerInfo.sampler = postProcess.linearSampler;

            /*VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = postProcess.postProcessUBO[frame].buffer;
            uboInfo.offset = 0;
            uboInfo.range = sizeof(PostProcessParams);*/

            std::array<VkWriteDescriptorSet, 4> writes{};
            
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = postProcess.compositeSets[frame];
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[0].descriptorCount = 1;
            writes[0].pImageInfo = &hdrInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = postProcess.compositeSets[frame];
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[1].descriptorCount = PostProcessResources::BLOOM_MIP_LEVELS;
            writes[1].pImageInfo = bloomInfos.data();

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = postProcess.compositeSets[frame];
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[2].descriptorCount = 1;
            writes[2].pImageInfo = &outputInfo;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = postProcess.compositeSets[frame];
            writes[3].dstBinding = 3;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[3].descriptorCount = 1;
            writes[3].pImageInfo = &samplerInfo;

            /*writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = postProcess.compositeSets[frame];
            writes[4].dstBinding = 4;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[4].descriptorCount = 1;
            writes[4].pBufferInfo = &uboInfo;*/

            vkUpdateDescriptorSets(device->logicalDevice, 4, writes.data(), 0, nullptr);
        }
    }
}

//void BlackHoleSim::createPostProcessUBO()
//{
//    postProcess.postProcessUBO.resize(MAX_CONCURRENT_FRAMES);
//    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
//        postProcess.postProcessUBO[i].create(
//            device->logicalDevice,
//            device->physicalDevice,
//            sizeof(PostProcessParams),
//            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
//            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
//        );
//        postProcess.postProcessUBO[i].map();
//    }
//}

void BlackHoleSim::postProcessWindowResizeRecreate()
{
    postProcess.hdrColorBuffer.destroy();
    postProcess.brightPassBuffer.destroy();

    for (auto& mip : postProcess.bloomMipChain) {
        mip.destroy();
    }

    createHDRColorBuffer();
    createBloomMipChain();
}

void BlackHoleSim::cleanupPostProcessResources()
{
    postProcess.hdrColorBuffer.destroy();
    postProcess.brightPassBuffer.destroy();
    postProcess.compositeOutputImage.destroy();

    for (auto& mip : postProcess.bloomMipChain) {
        mip.destroy();
    }
    postProcess.tempBlurBuffer.destroy();

    //for (auto& ubo : postProcess.postProcessUBO) ubo.destroy();

    vkDestroySampler(device->logicalDevice, postProcess.linearSampler, nullptr);
    vkDestroySampler(device->logicalDevice, postProcess.nearestSampler, nullptr);

    vkDestroyDescriptorSetLayout(device->logicalDevice, postProcess.brightExtractLayout, nullptr);
    vkDestroyDescriptorSetLayout(device->logicalDevice, postProcess.blurLayout, nullptr);
    vkDestroyDescriptorSetLayout(device->logicalDevice, postProcess.compositeLayout, nullptr);

    vkDestroyPipelineLayout(device->logicalDevice, postProcess.brightExtractPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device->logicalDevice, postProcess.blurPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device->logicalDevice, postProcess.compositePipelineLayout, nullptr);

    vkDestroyPipeline(device->logicalDevice, postProcess.brightExtractPipeline, nullptr);
    vkDestroyPipeline(device->logicalDevice, postProcess.blurHorizontalPipeline, nullptr);
    vkDestroyPipeline(device->logicalDevice, postProcess.blurVerticalPipeline, nullptr);
    vkDestroyPipeline(device->logicalDevice, postProcess.compositePipeline, nullptr);
}

void BlackHoleSim::performBloomPass(VkCommandBuffer cmd, uint32_t frameIndex)
{
    /*PostProcessParams ppParams{};
    ppParams.bloomThreshold = 3.0f;
    ppParams.bloomSoftKnee = 0.5f;
    ppParams.bloomIntensity = 0.4f;
    ppParams.exposure = 1.0f;
    ppParams.gamma = 1.0f;
    postProcess.postProcessUBO[currentFrame].copyTo(&ppParams, sizeof(PostProcessParams));*/

    // BRIGHT EXTRACT PASS
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postProcess.brightExtractPipeline);
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        postProcess.brightExtractPipelineLayout,
        0, 1,
        &postProcess.brightExtractSets[frameIndex],
        0, nullptr
    );

    BloomExtractParams bloomParams{};
    bloomParams.threshold = 1.0f;
    bloomParams.softKnee = 0.5;
    bloomParams.intensity = 0.6f;
    
    vkCmdPushConstants(cmd, postProcess.brightExtractPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(BloomExtractParams)), &bloomParams);

    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

    // bright extract write -> first blur read
    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.brightPassBuffer.image,
        VK_ACCESS_SHADER_WRITE_BIT, // bright extract wrote
        VK_ACCESS_SHADER_READ_BIT, // first blur will read
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );
    


    // Blur passes for each mip level
    for (int mip = 0; mip < PostProcessResources::BLOOM_MIP_LEVELS; mip++)
    {
        uint32_t mipWidth = std::max(1u, width >> mip);
        uint32_t mipHeight = std::max(1u, height >> mip);

        
        //ppParams.currentMipLevel = mip;
        //ppParams.blurDirection = 1; // Horizontal
        //ppParams.texelSizeX = 1.0f / float(mipWidth);
        //ppParams.texelSizeY = 1.0f / float(mipHeight);
        //postProcess.postProcessUBO[frameIndex].copyTo(&ppParams, sizeof(PostProcessParams));

        GaussianBlurParams blurParams{};
        blurParams.blurDirection = 1;
        blurParams.mipLevel = mip;
        blurParams.texelSizeX = 1.0f / float(mipWidth);
        blurParams.texelSizeY = 1.0f / float(mipHeight);

        vks::tools::insertImageMemoryBarrier(
            cmd,
            postProcess.tempBlurBuffer.image,
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        );


        // Horizontal blur
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postProcess.blurHorizontalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            postProcess.blurPipelineLayout, 0, 1,
            &postProcess.blurSets[frameIndex][mip * 2], 0, nullptr);

        vkCmdPushConstants(cmd, postProcess.blurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(GaussianBlurParams)), &blurParams);

        uint32_t groupX = (mipWidth + 15) / 16;
        uint32_t groupY = (mipHeight + 15) / 16;
        vkCmdDispatch(cmd, groupX, groupY, 1);

        vks::tools::insertImageMemoryBarrier(
            cmd,
            postProcess.tempBlurBuffer.image,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        );


        // Vertical blur
        //ppParams.blurDirection = 0; // vertical
        //postProcess.postProcessUBO[frameIndex].copyTo(&ppParams, sizeof(PostProcessParams));

        blurParams.blurDirection = 0;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postProcess.blurVerticalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            postProcess.blurPipelineLayout, 0, 1,
            &postProcess.blurSets[frameIndex][mip * 2 + 1], 0, nullptr);

        vkCmdPushConstants(cmd, postProcess.blurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(GaussianBlurParams)), &blurParams);

        groupX = (mipWidth + 15) / 16;
        groupY = (mipHeight + 15) / 16;
        vkCmdDispatch(cmd, groupX, groupY, 1);


        vks::tools::insertImageMemoryBarrier(
            cmd,
            postProcess.bloomMipChain[mip].image,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        );
    }
}

void BlackHoleSim::performFinalComposite(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t swapchainIndex)
{
    
    CompositeParams compositeParams{};
    compositeParams.exposure = 0.7f;
    compositeParams.gamma = 0.8f;

    // Run composite shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postProcess.compositePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        postProcess.compositePipelineLayout, 0, 1,
        &postProcess.compositeSets[frameIndex], 0, nullptr);

    vkCmdPushConstants(cmd, postProcess.compositePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(CompositeParams)), &compositeParams);

    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.compositeOutputImage.image,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::insertImageMemoryBarrier(
        cmd,
        swapchain->images[swapchainIndex],
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VkImageBlit blit{};
    blit.srcOffsets[1] = { (int32_t)width, (int32_t)height, 1 };
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstOffsets[1] = { (int32_t)width, (int32_t)height, 1 };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    vkCmdBlitImage(
        cmd,
        postProcess.compositeOutputImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapchain->images[swapchainIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit,
        VK_FILTER_LINEAR
    );
}


// synchronization
void BlackHoleSim::initializePPImageLayouts()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = device->graphicsCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device->logicalDevice, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &beginInfo));
    
    // transition storage image to be writable by the shader
    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.hdrColorBuffer.image,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT,  0, 1, 0, 1 }
    );

    // Initialize bright pass buffer for writing
    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.brightPassBuffer.image,
        0,                              // Was undefined
        VK_ACCESS_SHADER_WRITE_BIT,     // Will be written to
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    // Initialize all bloom mip levels for writing
    std::vector<VkImageMemoryBarrier> mipBarriers;
    for (int i = 0; i < PostProcessResources::BLOOM_MIP_LEVELS; i++) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = postProcess.bloomMipChain[i].image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        mipBarriers.push_back(barrier);
    }
    if (!mipBarriers.empty()) {
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(mipBarriers.size()), mipBarriers.data()
        );
    }

    // will be transitioned to write during ping pong
    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.tempBlurBuffer.image,
        0,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );


    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.compositeOutputImage.image,
        0,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(device->logicalDevice, &fenceInfo, nullptr, &fence));


    vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, fence);
    vkQueueWaitIdle(device->graphicsQueue);

    vkWaitForFences(device->logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device->logicalDevice, fence, nullptr);
    vkFreeCommandBuffers(device->logicalDevice, device->graphicsCommandPool, 1, &cmd);
}
void BlackHoleSim::ppImageLayoutTransitionNextFrame(VkCommandBuffer cmd)
{
    // hdr back to write
    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.hdrColorBuffer.image,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.brightPassBuffer.image,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    std::vector<VkImageMemoryBarrier> mipBarriers;
    for (int i = 0; i < PostProcessResources::BLOOM_MIP_LEVELS; i++) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = postProcess.bloomMipChain[i].image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        mipBarriers.push_back(barrier);
    }
    if (!mipBarriers.empty()) {
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(mipBarriers.size()), mipBarriers.data()
        );
    }

    vks::tools::insertImageMemoryBarrier(
        cmd,
        postProcess.compositeOutputImage.image,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );
}