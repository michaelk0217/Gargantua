#include "UIOverlay.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <glm/glm.hpp>

#include "Window.h"
#include "VulkanDevice.h"
#include "VulkanSwapchain.h"
#include "VulkanStructures.h"

#include "VulkanTools.h"

UIOverlay::UIOverlay(Window& window, VulkanDevice& device, VulkanSwapchain& swapchain) : m_device(device), m_swapchain(swapchain)
{
    VkDescriptorPoolSize pool_sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    poolCI.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    poolCI.pPoolSizes = pool_sizes;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device.logicalDevice, &poolCI, nullptr, &descriptorPool));

    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window.getGlfwWindow(), true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_4;
    initInfo.Instance = device.instance;
    initInfo.PhysicalDevice = device.physicalDevice;
    initInfo.Device = device.logicalDevice;
    initInfo.QueueFamily = device.familyIndices.graphicsFamily.value();
    initInfo.Queue = device.graphicsQueue;
    initInfo.DescriptorPool = descriptorPool;
    initInfo.MinImageCount = swapchain.imageViews.size();
    initInfo.ImageCount = swapchain.imageViews.size();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.UseDynamicRendering = true;

    VkFormat colorAttacmentFormat = swapchain.colorFormat;

    VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
    pipelineRenderingCI.colorAttachmentCount = 1;
    pipelineRenderingCI.pColorAttachmentFormats = &colorAttacmentFormat;
    pipelineRenderingCI.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    pipelineRenderingCI.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    initInfo.PipelineRenderingCreateInfo = pipelineRenderingCI;

    ImGui_ImplVulkan_Init(&initInfo);
}

UIOverlay::~UIOverlay()
{
    vkDeviceWaitIdle(m_device.logicalDevice);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(m_device.logicalDevice, descriptorPool, nullptr);
}

void UIOverlay::newFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UIOverlay::buildUI(UIPacket& uiPacket)
{
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.12f, 0.75f));
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 320, 20), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_Always);
    if (ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav))
    {
        ImGui::Text("Performace");
        ImGui::Separator();

        char overlay_text[32];
        sprintf_s(overlay_text, "FPS: %.1f", 1.0f / uiPacket.deltaTime);
        ImGui::PlotLines("##framerate", uiPacket.frameHistory.data(), static_cast<int>(uiPacket.frameHistory.size()), 0, overlay_text, 0.0f, 3000.0f, ImVec2(0, 50));
        ImGui::Text("Frame Time: %.7f ms", uiPacket.deltaTime);

        ImGui::Text("Camera Debug");
        ImGui::Separator();

        ImGui::Text("Direction: (%.2f, %.2f, %.2f)",
            uiPacket.cameraDirection.x,
            uiPacket.cameraDirection.y,
            uiPacket.cameraDirection.z);

        /*glm::vec3 centerColor = uiPacket.cameraDirection * 0.5f + 0.5f;

        ImGui::Text("Center Color (RGB): (%.2f, %.2f, %.2f)",
            centerColor.r,
            centerColor.g,
            centerColor.b);

        ImGui::ColorButton("Center Color Swatch",
            ImVec4(centerColor.r, centerColor.g, centerColor.b, 1.0f),
            ImGuiColorEditFlags_NoTooltip,
            ImVec2(ImGui::GetContentRegionAvail().x, 20));*/
    }
    ImGui::End();
    //ImGui::PopStyleColor();

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(400, 350), ImGuiCond_Always);
    if (ImGui::Begin("Black Hole Controls", nullptr, ImGuiWindowFlags_None))
    {
        ImGui::Text("Gargantua Parameters");
        ImGui::Separator();

        ImGui::Text("Black hole Mass");
        ImGui::SliderFloat("##Mass", &uiPacket.blackHoleMass, 0.1f, 5.0f, "%0.1f");

        ImGui::Text("Spin Parameter");
        
        ImGui::SliderFloat("##Spin", &uiPacket.blackHoleSpin, 0.0f, 0.999f, "%.3f");
        //if (ImGui::Button("Max -Spin")) uiPacket.blackHoleSpin = -0.999f;
        //ImGui::SameLine();
        if (ImGui::Button("Max Spin")) uiPacket.blackHoleSpin = 0.999f;

        ImGui::Text("Ray Marching Quality");
        ImGui::SliderInt("##MaxSteps", &uiPacket.maxSteps, 10, 500, "%d steps");
        ImGui::SliderFloat("##StepSize", &uiPacket.stepSize, 0.01f, 1.0f, "Step: %.2f");

        ImGui::Text("Background");
        const char* backgroundTypes[] = { "Procedural", "HDR Environment", "Color Spectrum"};
        ImGui::Combo("##Background", &uiPacket.backgroundType, backgroundTypes, IM_ARRAYSIZE(backgroundTypes));

        ImGui::Text("Geodesic Computation");
        const char* geodesicComps[] = { "Simple", "4th Order RK" };
        ImGui::Combo("##Compute Type", &uiPacket.geodesicType, geodesicComps, IM_ARRAYSIZE(geodesicComps));

        ImGui::Text("Accretion Disk Parameters");
        ImGui::Separator();
        if (ImGui::Button("Toggle Disk")) {
            uiPacket.diskEnable = !uiPacket.diskEnable;
        }

        // presets
        ImGui::Separator();
        if (ImGui::Button("Interstellar Preset")) {
            uiPacket.blackHoleMass = 1.0f;
            uiPacket.blackHoleSpin = 0.6f;
            uiPacket.maxSteps = 150;
            uiPacket.stepSize = 0.15f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Defaults")) {
            uiPacket.blackHoleMass = 1.0f;
            uiPacket.blackHoleSpin = 0.0f;
            uiPacket.maxSteps = 100;
            uiPacket.stepSize = 0.2f;
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
}

void UIOverlay::render(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height)
{
    ImGui::Render();

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_swapchain.imageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = { 0, 0, width, height };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRendering(commandBuffer);
}

void UIOverlay::update_frame_history(std::vector<float>& frame_history, float framerate)
{
    if (!frame_history.empty())
    {
        frame_history.erase(frame_history.begin());
    }
    frame_history.push_back(framerate);
}
