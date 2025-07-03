package club.doki7.rkt.vk.common;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.vulkan.enumtype.VkFormat;

public enum PixelFormat {
    COLOR_ATTACHMENT_DEFAULT(VkFormat.R8G8B8A8_SRGB, 4),
    TEXTURE_DEFAULT(VkFormat.R8G8B8A8_SRGB, 4),
    DEPTH_DEFAULT(VkFormat.D32_SFLOAT, 4),
    DEPTH_STENCIL_DEFAULT(VkFormat.D24_UNORM_S8_UINT, 4),
    SWAPCHAIN_DEFAULT(VkFormat.B8G8R8A8_SRGB, 4),
    UINT8(VkFormat.R8_UINT, 1),
    UINT32(VkFormat.R32_UINT, 4),
    SFLOAT32(VkFormat.R32_SFLOAT, 4);

    public final @EnumType(VkFormat.class) int format;
    public final int bytesPerPixel;

    PixelFormat(@EnumType(VkFormat.class) int format, int bytesPerPixel) {
        this.format = format;
        this.bytesPerPixel = bytesPerPixel;
    }
}
