package club.doki7.rkt.vk.resc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.vulkan.enumtype.VkImageType;

public enum ImageKind {
    _1D, _2D, _3D;

    public static ImageKind from(@EnumType(VkImageType.class) int imageType) {
        return switch (imageType) {
            case VkImageType._1D -> _1D;
            case VkImageType._2D -> _2D;
            case VkImageType._3D -> _3D;
            default -> throw new IllegalArgumentException("无效的图像类型: " + imageType);
        };
    }
}
