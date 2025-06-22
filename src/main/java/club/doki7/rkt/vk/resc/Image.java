package club.doki7.rkt.vk.resc;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vma.bitmask.VmaAllocationCreateFlags;
import club.doki7.vma.datatype.VmaAllocationCreateInfo;
import club.doki7.vma.datatype.VmaAllocationInfo;
import club.doki7.vma.enumtype.VmaMemoryUsage;
import club.doki7.vma.handle.VmaAllocation;
import club.doki7.vulkan.bitmask.VkImageCreateFlags;
import club.doki7.vulkan.bitmask.VkImageUsageFlags;
import club.doki7.vulkan.bitmask.VkSampleCountFlags;
import club.doki7.vulkan.datatype.VkImageCreateInfo;
import club.doki7.vulkan.enumtype.*;
import club.doki7.vulkan.handle.VkImage;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;

public final class Image implements AutoCloseable {
    public final VkImage handle;
    public final ImageKind kind;
    public final @EnumType(VkFormat.class) int format;
    public final int width;
    public final int height;
    public final int depth;
    public final int mipLevels;
    public final @Nullable BytePtr mapped;

    public static Image create(
            RenderContext cx,
            @Bitmask(VkImageCreateFlags.class) int flags,
            @EnumType(VkImageType.class) int imageType,
            @EnumType(VkFormat.class) int format,
            int width,
            int height,
            int depth,
            int arrayLayers,
            int mipLevels,
            @Bitmask(VkSampleCountFlags.class) int samples,
            @Bitmask(VkImageUsageFlags.class) int usage,
            @Bitmask(VmaAllocationCreateFlags.class) int allocationFlags,
            int @Nullable [] sharedQueueFamilyIndices,
            boolean local
    ) throws VulkanException {
        boolean initiallyMapped = (allocationFlags & VmaAllocationCreateFlags.MAPPED) != 0;

        try (Arena arena = Arena.ofConfined()) {
            VkImageCreateInfo imageCreateInfo = VkImageCreateInfo.allocate(arena)
                    .flags(flags)
                    .imageType(imageType)
                    .format(format)
                    .extent(extent -> extent.width(width).height(height).depth(depth))
                    .mipLevels(mipLevels)
                    .arrayLayers(arrayLayers)
                    .samples(samples)
                    .tiling(initiallyMapped ? VkImageTiling.LINEAR : VkImageTiling.OPTIMAL)
                    .usage(usage);
            if (sharedQueueFamilyIndices != null) {
                imageCreateInfo.sharingMode(VkSharingMode.CONCURRENT)
                        .queueFamilyIndexCount(sharedQueueFamilyIndices.length)
                        .pQueueFamilyIndices(IntPtr.allocate(arena, sharedQueueFamilyIndices));
            } else {
                imageCreateInfo.sharingMode(VkSharingMode.EXCLUSIVE);
            }
            VmaAllocationCreateInfo allocationCreateInfo = VmaAllocationCreateInfo.allocate(arena)
                    .usage(VmaMemoryUsage.AUTO)
                    .flags(allocationFlags);
            @Nullable VmaAllocationInfo allocationInfo = initiallyMapped
                    ? VmaAllocationInfo.allocate(arena)
                    : null;

            VkImage.Ptr pImage = VkImage.Ptr.allocate(arena);
            VmaAllocation.Ptr pAllocation = VmaAllocation.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.vma.createImage(
                    cx.vmaAllocator,
                    imageCreateInfo,
                    allocationCreateInfo,
                    pImage,
                    pAllocation,
                    allocationInfo
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 图像");
            }

            VkImage handle = pImage.read();
            VmaAllocation allocation = pAllocation.read();
            @Nullable BytePtr mapped;
            if (allocationInfo != null) {
                long mappedSize = width * height * depth * arrayLayers * (long) formatSize(format);
                mapped = new BytePtr(allocationInfo.pMappedData().reinterpret(mappedSize));
            } else {
                mapped = null;
            }
            return new Image(
                    handle,
                    ImageKind.from(imageType),
                    format,
                    width,
                    height,
                    depth,
                    mipLevels,
                    mapped,
                    allocation,
                    cx,
                    local
            );
        }
    }

    public static Image createStaticTexture2D(
            RenderContext cx,
            @EnumType(VkFormat.class) int format,
            int width,
            int height,
            int mipLevels
    ) {
        try {
            return create(
                    cx,
                    /*flags=*/0,
                    VkImageType._2D,
                    format,
                    width,
                    height,
                    /*depth=*/1,
                    /*arrayLayers=*/1,
                    mipLevels,
                    VkSampleCountFlags._1,
                    VkImageUsageFlags.SAMPLED | VkImageUsageFlags.TRANSFER_DST,
                    /*allocationFlags=*/0,
                    /*sharedQueueFamilyIndices=*/null,
                    /*local=*/false
            );
        } catch (VulkanException e) {
            throw new RuntimeException("无法创建 Vulkan 纹理图像", e);
        }
    }

    public static Image createDynamicTexture2D(
            RenderContext cx,
            @EnumType(VkFormat.class) int format,
            int width,
            int height
    ) {
        try {
            return create(
                    cx,
                    /*flags=*/0,
                    VkImageType._2D,
                    format,
                    width,
                    height,
                    /*depth=*/1,
                    /*arrayLayers=*/1,
                    /*mipLevels=*/1,
                    VkSampleCountFlags._1,
                    VkImageUsageFlags.SAMPLED,
                    VmaAllocationCreateFlags.HOST_ACCESS_RANDOM | VmaAllocationCreateFlags.MAPPED,
                    /*sharedQueueFamilyIndices=*/null,
                    /*local=*/false
            );
        } catch (VulkanException e) {
            throw new RuntimeException("无法创建 Vulkan 动态纹理图像", e);
        }
    }

    public static final class AttachmentAccess {
        public boolean sampling = true;
        public boolean graphicsPipelineWrite = true;
        public boolean computePipelineAccess = true;
        public boolean transferSource = true;
        public boolean transferDest = true;

        public @Bitmask(VkImageUsageFlags.class) int computeUsageFlags() {
            int usage = 0;
            if (sampling) {
                usage |= VkImageUsageFlags.SAMPLED;
            }
            if (graphicsPipelineWrite) {
                usage |= VkImageUsageFlags.COLOR_ATTACHMENT;
            }
            if (computePipelineAccess) {
                usage |= VkImageUsageFlags.STORAGE;
            }
            if (transferSource) {
                usage |= VkImageUsageFlags.TRANSFER_SRC;
            }
            if (transferDest) {
                usage |= VkImageUsageFlags.TRANSFER_DST;
            }
            return usage;
        }
    }

    public static Image createAttachment2D(
            RenderContext cx,
            @EnumType(VkFormat.class) int format,
            int width,
            int height,
            @Nullable AttachmentAccess access
    ) {
        @Bitmask(VkImageUsageFlags.class) int usage;
        if (access == null) {
            usage = VkImageUsageFlags.COLOR_ATTACHMENT
                    | VkImageUsageFlags.SAMPLED
                    | VkImageUsageFlags.STORAGE
                    | VkImageUsageFlags.TRANSFER_SRC
                    | VkImageUsageFlags.TRANSFER_DST;
        } else {
            usage = access.computeUsageFlags();
        }

        try {
            return create(
                    cx,
                    /*flags=*/0,
                    VkImageType._2D,
                    format,
                    width,
                    height,
                    /*depth=*/1,
                    /*arrayLayers=*/1,
                    /*mipLevels=*/1,
                    VkSampleCountFlags._1,
                    usage,
                    /*allocationFlags=*/0,
                    /*sharedQueueFamilyIndices=*/null,
                    /*local=*/false
            );
        } catch (VulkanException e) {
            throw new RuntimeException("无法创建 Vulkan 附件图像", e);
        }
    }

    public static Image createDepthStencil(
            RenderContext cx,
            @EnumType(VkFormat.class) int format,
            int width,
            int height
    ) {
        try {
            return create(
                    cx,
                    /*flags=*/0,
                    VkImageType._2D,
                    format,
                    width,
                    height,
                    /*depth=*/1,
                    /*arrayLayers=*/1,
                    /*mipLevels=*/1,
                    VkSampleCountFlags._1,
                    VkImageUsageFlags.DEPTH_STENCIL_ATTACHMENT,
                    /*allocationFlags=*/0,
                    /*sharedQueueFamilyIndices=*/null,
                    /*local=*/false
            );
        } catch (VulkanException e) {
            throw new RuntimeException("无法创建 Vulkan 深度模板图像", e);
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private Image(
            VkImage handle,
            ImageKind kind,
            @EnumType(VkFormat.class) int format,
            int width,
            int height,
            int depth,
            int mipLevels,
            @Nullable BytePtr mapped,
            VmaAllocation allocation,
            RenderContext context,
            boolean local
    ) {
        this.handle = handle;
        this.kind = kind;
        this.format = format;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.mipLevels = mipLevels;
        this.mapped = mapped;

        IDisposeOnContext d = cx -> cx.vma.destroyImage(cx.vmaAllocator, handle, allocation);
        this.cleanable = context.registerCleanup(this, d, local);
    }

    private final Cleaner.Cleanable cleanable;

    private static int formatSize(@EnumType(VkFormat.class) int format) {
        return switch (format) {
            case VkFormat.R8_UNORM, VkFormat.R8_SNORM, VkFormat.R8_UINT, VkFormat.R8_SINT -> 1;
            case VkFormat.R8G8_UNORM, VkFormat.R8G8_SNORM, VkFormat.R8G8_UINT, VkFormat.R8G8_SINT,
                 VkFormat.R8G8_SRGB -> 2;
            case VkFormat.R8G8B8_UNORM, VkFormat.R8G8B8_SNORM, VkFormat.R8G8B8_UINT, VkFormat.R8G8B8_SINT,
                 VkFormat.R8G8B8_SRGB -> 3;
            case VkFormat.R8G8B8A8_UNORM, VkFormat.R8G8B8A8_SNORM, VkFormat.R8G8B8A8_UINT,
                 VkFormat.R8G8B8A8_SINT, VkFormat.R8G8B8A8_SRGB, VkFormat.B8G8R8A8_UNORM,
                 VkFormat.B8G8R8A8_SNORM, VkFormat.B8G8R8A8_UINT, VkFormat.B8G8R8A8_SINT,
                 VkFormat.R32_SFLOAT, VkFormat.R32_UINT, VkFormat.R32_SINT,
                 VkFormat.D24_UNORM_S8_UINT, VkFormat.D32_SFLOAT -> 4;
            default -> throw new IllegalArgumentException("不支持的图像格式：" + VkFormat.explain(format));
        };
    }
}
