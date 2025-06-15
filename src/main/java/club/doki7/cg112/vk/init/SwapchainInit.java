package club.doki7.cg112.vk.init;

import club.doki7.cg112.exc.RenderException;
import club.doki7.cg112.exc.VulkanException;
import club.doki7.cg112.vk.RenderConfig;
import club.doki7.cg112.vk.RenderContext;
import club.doki7.cg112.vk.Swapchain;
import club.doki7.ffm.NativeLayout;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.*;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.enumtype.*;
import club.doki7.vulkan.handle.VkImage;
import club.doki7.vulkan.handle.VkImageView;
import club.doki7.vulkan.handle.VkSwapchainKHR;

import java.lang.foreign.Arena;
import java.util.Objects;

public final class SwapchainInit {
    private final RenderContext cx;

    private VkSurfaceFormatKHR surfaceFormat;
    private @EnumType(VkPresentModeKHR.class) int presentMode;
    private VkExtent2D swapExtent;
    private @EnumType(VkSurfaceTransformFlagsKHR.class) int currentTransform;

    private int imageCount;
    private VkSwapchainKHR vkSwapchain;
    private VkImage.Ptr pSwapchainImages;
    private VkImageView.Ptr pSwapchainImageViews;

    public SwapchainInit(RenderContext cx) {
        this.cx = cx;
    }

    public Swapchain init(int width, int height) throws RenderException {
        try (Arena arena = Arena.ofConfined()) {
            SwapchainSupportDetails swapchainSupportDetails = querySwapchainSupportDetails(arena);

            chooseSwapchainSurfaceFormat(swapchainSupportDetails.formats());
            chooseSwapchainPresentMode(swapchainSupportDetails.presentModes());
            chooseSwapExtent(swapchainSupportDetails.capabilities(), width, height);

            currentTransform = swapchainSupportDetails.capabilities.currentTransform();
            imageCount = swapchainSupportDetails.capabilities().minImageCount() + 1;
            if (swapchainSupportDetails.capabilities().maxImageCount() > 0 &&
                imageCount > swapchainSupportDetails.capabilities().maxImageCount()) {
                imageCount = swapchainSupportDetails.capabilities().maxImageCount();
            }

            createSwapchain();
            getSwapchainImages();
            createSwapchainImageViews();
        } catch (Throwable e) {
            cleanup();
            throw e;
        }

        return new Swapchain(
                cx,
                surfaceFormat,
                swapExtent,

                vkSwapchain,
                pSwapchainImages,
                pSwapchainImageViews
        );
    }

    private void createSwapchain() throws RenderException {
        try (Arena arena = Arena.ofConfined()) {
            VkSwapchainCreateInfoKHR swapchainCreateInfo = VkSwapchainCreateInfoKHR.allocate(arena)
                    .surface(cx.surface)
                    .minImageCount(imageCount)
                    .imageFormat(surfaceFormat.format())
                    .imageColorSpace(surfaceFormat.colorSpace())
                    .imageExtent(swapExtent)
                    .imageArrayLayers(1)
                    .imageUsage(VkImageUsageFlags.COLOR_ATTACHMENT)
                    .preTransform(currentTransform)
                    .compositeAlpha(VkCompositeAlphaFlagsKHR.OPAQUE)
                    .presentMode(presentMode)
                    .clipped(VkConstants.TRUE);

            if (cx.graphicsQueueFamilyIndex != cx.presentQueueFamilyIndex) {
                swapchainCreateInfo
                        .imageSharingMode(VkSharingMode.CONCURRENT)
                        .pQueueFamilyIndices(IntPtr.allocateV(
                                arena,
                                cx.graphicsQueueFamilyIndex,
                                cx.presentQueueFamilyIndex
                        ));
            } else {
                swapchainCreateInfo.imageSharingMode(VkSharingMode.EXCLUSIVE);
            }

            VkSwapchainKHR.Ptr pSwapchain = VkSwapchainKHR.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.dCmd.createSwapchainKHR(
                    cx.device,
                    swapchainCreateInfo,
                    null,
                    pSwapchain
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 交换链");
            }
            vkSwapchain = Objects.requireNonNull(pSwapchain.read());
        }
    }

    private void getSwapchainImages() throws RenderException {
        pSwapchainImages = VkImage.Ptr.allocate(cx.prefabArena, imageCount);

        try (Arena arena = Arena.ofConfined()) {
            IntPtr pImageCount = IntPtr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.getSwapchainImagesKHR(cx.device, vkSwapchain, pImageCount, null);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 交换链图像数量, 错误代码");
            }
            int resultingImageCount = pImageCount.read();
            assert resultingImageCount == imageCount : String.format(
                    "实际创建的交换链图像数量 (%d) 与预设数量 (%d) 不匹配",
                    imageCount,
                    resultingImageCount
            );

            result = cx.dCmd.getSwapchainImagesKHR(cx.device, vkSwapchain, pImageCount, pSwapchainImages);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 交换链图像, 错误代码");
            }
        }
    }

    private void createSwapchainImageViews() throws RenderException {
        pSwapchainImageViews = VkImageView.Ptr.allocate(cx.prefabArena, pSwapchainImages.size());

        try (Arena arena = Arena.ofConfined()) {
            for (long i = 0; i < pSwapchainImages.size(); i++) {
                VkImage image = pSwapchainImages.read(i);
                VkImageView.Ptr pImageView = pSwapchainImageViews.offset(i);

                VkImageViewCreateInfo imageViewCreateInfo = VkImageViewCreateInfo.allocate(arena)
                        .image(image)
                        .viewType(VkImageViewType._2D)
                        .format(surfaceFormat.format())
                        .subresourceRange(range -> range
                                .aspectMask(VkImageAspectFlags.COLOR)
                                .baseMipLevel(0)
                                .levelCount(1)
                                .baseArrayLayer(0)
                                .layerCount(1));
                @EnumType(VkResult.class) int result = cx.dCmd.createImageView(
                        cx.device,
                        imageViewCreateInfo,
                        null,
                        pImageView
                );
                if (result != VkResult.SUCCESS) {
                    throw new RenderException("无法创建 Vulkan 图像视图, 错误代码: " + VkResult.explain(result));
                }
            }
        }
    }

    private void cleanup() {
        if (pSwapchainImageViews != null) {
            for (VkImageView imageView : pSwapchainImageViews) {
                if (imageView != null) {
                    cx.dCmd.destroyImageView(cx.device, imageView, null);
                }
            }
        }

        if (vkSwapchain != null) {
            cx.dCmd.destroySwapchainKHR(cx.device, vkSwapchain, null);
        }
    }

    private record SwapchainSupportDetails(
            VkSurfaceCapabilitiesKHR capabilities,
            VkSurfaceFormatKHR.Ptr formats,
            @EnumType(VkPresentModeKHR.class) IntPtr presentModes
    ) {}

    private SwapchainSupportDetails querySwapchainSupportDetails(Arena arena) throws RenderException {
        VkSurfaceCapabilitiesKHR surfaceCapabilities = VkSurfaceCapabilitiesKHR.allocate(arena);
        @EnumType(VkResult.class) int result = cx.iCmd.getPhysicalDeviceSurfaceCapabilitiesKHR(
                cx.physicalDevice,
                cx.surface,
                surfaceCapabilities
        );
        if (result != VkResult.SUCCESS) {
            throw new VulkanException(result, "无法获取 Vulkan 表面能力");
        }

        try (Arena localArena = Arena.ofConfined()) {
            IntPtr pCount = IntPtr.allocate(localArena);
            result = cx.iCmd.getPhysicalDeviceSurfaceFormatsKHR(cx.physicalDevice, cx.surface, pCount, null);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 表面格式数量");
            }

            int formatCount = pCount.read();
            VkSurfaceFormatKHR.Ptr surfaceFormats = VkSurfaceFormatKHR.allocate(arena, formatCount);
            result = cx.iCmd.getPhysicalDeviceSurfaceFormatsKHR(
                    cx.physicalDevice,
                    cx.surface,
                    pCount,
                    surfaceFormats
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 表面格式");
            }

            result = cx.iCmd.getPhysicalDeviceSurfacePresentModesKHR(
                    cx.physicalDevice,
                    cx.surface,
                    pCount,
                    null
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 表面呈现模式数量");
            }

            int presentModeCount = pCount.read();
            IntPtr presentModes = IntPtr.allocate(arena, presentModeCount);
            result = cx.iCmd.getPhysicalDeviceSurfacePresentModesKHR(
                    cx.physicalDevice,
                    cx.surface,
                    pCount,
                    presentModes
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 表面呈现模式");
            }

            return new SwapchainSupportDetails(surfaceCapabilities, surfaceFormats, presentModes);
        }
    }

    private void chooseSwapchainSurfaceFormat(VkSurfaceFormatKHR.Ptr formats) {
        surfaceFormat = VkSurfaceFormatKHR.allocate(cx.prefabArena);
        @EnumType(VkFormat.class) int preferredFormat = VkFormat.B8G8R8A8_SRGB;
        for (VkSurfaceFormatKHR format : formats) {
            if (format.format() == preferredFormat &&
                format.colorSpace() == VkColorSpaceKHR.SRGB_NONLINEAR) {
                surfaceFormat.segment().copyFrom(format.segment());
            }
        }
        surfaceFormat.segment().copyFrom(formats.at(0).segment());
    }

    private void chooseSwapchainPresentMode(@EnumType(VkPresentModeKHR.class) IntPtr presentModes) {
        RenderConfig.VSync vsync = cx.config.vsync;

        if (vsync == RenderConfig.VSync.ON) {
            presentMode = VkPresentModeKHR.FIFO;
            return;
        }

        for (int i = 0; i < presentModes.size(); i++) {
            @EnumType(VkPresentModeKHR.class) int mode = presentModes.read(i);
            if (mode == VkPresentModeKHR.MAILBOX) {
                presentMode = mode;
                return;
            }
        }

        if (vsync == RenderConfig.VSync.PREFER_OFF) {
            for (int i = 0; i < presentModes.size(); i++) {
                @EnumType(VkPresentModeKHR.class) int mode = presentModes.read(i);
                if (mode == VkPresentModeKHR.IMMEDIATE) {
                    presentMode = mode;
                    return;
                }
            }
        }

        presentMode = VkPresentModeKHR.FIFO;
    }

    private void chooseSwapExtent(
            VkSurfaceCapabilitiesKHR capabilities,
            int width,
            int height
    ) {
        swapExtent = VkExtent2D.allocate(cx.prefabArena);
        if (capabilities.currentExtent().width() != NativeLayout.UINT32_MAX) {
            swapExtent.segment().copyFrom(capabilities.currentExtent().segment());
        } else {
            swapExtent.width(Math.max(
                    capabilities.minImageExtent().width(),
                    Math.min(capabilities.maxImageExtent().width(), width)
            ));
            swapExtent.height(Math.max(
                    capabilities.minImageExtent().height(),
                    Math.min(capabilities.maxImageExtent().height(), height)
            ));
        }
    }
}
