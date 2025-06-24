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
import club.doki7.vulkan.bitmask.VkImageUsageFlags;
import club.doki7.vulkan.bitmask.VkMemoryPropertyFlags;
import club.doki7.vulkan.bitmask.VkSampleCountFlags;
import club.doki7.vulkan.datatype.VkImageCreateInfo;
import club.doki7.vulkan.enumtype.*;
import club.doki7.vulkan.handle.VkImage;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.Collections;
import java.util.Set;

public final class Image implements AutoCloseable {
    public enum Dimension {
        DIM_1D(VkImageType._1D),
        DIM_2D(VkImageType._2D),
        DIM_3D(VkImageType._3D);

        public final @EnumType(VkImageType.class) int imageType;

        Dimension(@EnumType(VkImageType.class) int imageType) {
            this.imageType = imageType;
        }
    }

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

    public enum Usage {
        TEXTURE(VkImageUsageFlags.SAMPLED),
        COLOR_ATTACHMENT(VkImageUsageFlags.COLOR_ATTACHMENT),
        DEPTH_STENCIL_ATTACHMENT(VkImageUsageFlags.DEPTH_STENCIL_ATTACHMENT),
        STORAGE(VkImageUsageFlags.STORAGE),
        TRANSFER_SRC(VkImageUsageFlags.TRANSFER_SRC),
        TRANSFER_DST(VkImageUsageFlags.TRANSFER_DST);

        @Bitmask(VkImageUsageFlags.class) final int flag;

        Usage(@Bitmask(VkImageUsageFlags.class) int flag) {
            this.flag = flag;
        }
    }

    public static final class Options {
        public final Set<Usage> usages;
        public final boolean mapped;
        public final boolean coherent;
        public final Set<Integer> sharedQueueFamilyIndices;

        final @Bitmask(VkImageUsageFlags.class) int usageFlags;
        final @Bitmask(VmaAllocationCreateFlags.class) int allocationCreateFlags;
        final @Bitmask(VkMemoryPropertyFlags.class) int memoryPropertyFlags;
        final @EnumType(VkImageTiling.class) int tiling;

        Options(
                Set<Usage> usages,
                boolean mapped,
                boolean coherent,
                Set<Integer> sharedQueueFamilyIndices
        ) {
            this.usages = Collections.unmodifiableSet(usages);
            this.mapped = mapped;
            this.coherent = coherent;
            this.sharedQueueFamilyIndices = Collections.unmodifiableSet(sharedQueueFamilyIndices);

            @Bitmask(VkImageUsageFlags.class) int usageFlags = 0;
            for (Usage usage : usages) {
                usageFlags |= usage.flag;
            }
            this.usageFlags = usageFlags;

            this.allocationCreateFlags = mapped
                    ? VmaAllocationCreateFlags.HOST_ACCESS_RANDOM | VmaAllocationCreateFlags.MAPPED
                    : 0;

            this.memoryPropertyFlags = coherent
                    ? VkMemoryPropertyFlags.HOST_COHERENT
                    : 0;

            this.tiling = mapped
                    ? VkImageTiling.LINEAR
                    : VkImageTiling.OPTIMAL;
        }
    }

    public static final class OptionsInit {
        public Set<Usage> usages;
        public boolean mapped;
        public boolean coherent;
        public Set<Integer> sharedQueueFamilyIndices;

        public OptionsInit() {
            this.usages = Collections.emptySet();
            this.mapped = false;
            this.coherent = false;
            this.sharedQueueFamilyIndices = Set.of();
        }

        public Options build() {
            if (coherent && !mapped) {
                throw new IllegalStateException("无效的参数组合：若指定了 coherent，则必须指定 mapped");
            }

            if (usages.isEmpty()) {
                throw new IllegalStateException("必须指定至少一个使用场景");
            }

            return new Options(
                    usages,
                    mapped,
                    coherent,
                    sharedQueueFamilyIndices
            );
        }

        public static OptionsInit texturePreset() {
            OptionsInit options = new OptionsInit();
            options.usages = Set.of(Usage.TEXTURE, Usage.TRANSFER_DST);
            options.mapped = false;
            options.coherent = false;
            return options;
        }

        public static OptionsInit colorAttachmentPreset() {
            OptionsInit options = new OptionsInit();
            options.usages = Set.of(Usage.COLOR_ATTACHMENT, Usage.TEXTURE);
            options.mapped = false;
            options.coherent = false;
            return options;
        }

        public static OptionsInit depthStencilPreset() {
            OptionsInit options = new OptionsInit();
            options.usages = Set.of(Usage.DEPTH_STENCIL_ATTACHMENT, Usage.TEXTURE);
            options.mapped = false;
            options.coherent = false;
            return options;
        }

        public static OptionsInit storagePreset() {
            OptionsInit options = new OptionsInit();
            options.usages = Set.of(Usage.STORAGE, Usage.TEXTURE);
            options.mapped = false;
            options.coherent = false;
            return options;
        }
    }

    public final VkImage handle;
    public final Dimension dimension;
    public final PixelFormat pixelFormat;
    public final int width;
    public final int height;
    public final int depth;
    public final int arrayLayers;
    public final int mipLevels;
    public final Options options;
    public final @Nullable BytePtr mapped;

    public static Image create(
            RenderContext cx,
            Dimension dimension,
            PixelFormat pixelFormat,
            int width,
            int height,
            int depth,
            int arrayLayers,
            int mipLevels,
            int sampleCount,
            Options options,
            boolean local
    ) throws VulkanException {
        @EnumType(VkSampleCountFlags.class) int samples = switch (sampleCount) {
            case 1 -> VkSampleCountFlags._1;
            case 2 -> VkSampleCountFlags._2;
            case 4 -> VkSampleCountFlags._4;
            case 8 -> VkSampleCountFlags._8;
            case 16 -> VkSampleCountFlags._16;
            case 32 -> VkSampleCountFlags._32;
            case 64 -> VkSampleCountFlags._64;
            default -> throw new IllegalArgumentException("无效的样本数：" + sampleCount);
        };

        try (Arena arena = Arena.ofConfined()) {
            VkImageCreateInfo imageCreateInfo = VkImageCreateInfo.allocate(arena)
                    .flags(0)
                    .imageType(dimension.imageType)
                    .format(pixelFormat.format)
                    .extent(extent -> extent.width(width).height(height).depth(depth))
                    .mipLevels(mipLevels)
                    .arrayLayers(arrayLayers)
                    .samples(samples)
                    .tiling(options.tiling)
                    .usage(options.usageFlags);
            if (!options.sharedQueueFamilyIndices.isEmpty()) {
                IntPtr pSharedQueueFamilyIndices =
                        IntPtr.allocate(arena, options.sharedQueueFamilyIndices.size());
                int index = 0;
                for (int queueFamilyIndex : options.sharedQueueFamilyIndices) {
                    pSharedQueueFamilyIndices.write(index, queueFamilyIndex);
                    index += 1;
                }

                imageCreateInfo.sharingMode(VkSharingMode.CONCURRENT)
                        .pQueueFamilyIndices(pSharedQueueFamilyIndices)
                        .queueFamilyIndexCount(options.sharedQueueFamilyIndices.size());
            } else {
                imageCreateInfo.sharingMode(VkSharingMode.EXCLUSIVE);
            }

            VmaAllocationCreateInfo allocationCreateInfo = VmaAllocationCreateInfo.allocate(arena)
                    .usage(VmaMemoryUsage.AUTO)
                    .flags(options.allocationCreateFlags)
                    .requiredFlags(options.memoryPropertyFlags);
            VmaAllocationInfo allocationInfo = VmaAllocationInfo.allocate(arena);

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
            if (options.mapped) {
                long mappedSize = (long) width * height * depth * arrayLayers * pixelFormat.bytesPerPixel;
                mapped = new BytePtr(allocationInfo.pMappedData().reinterpret(mappedSize));
            } else {
                mapped = null;
            }

            return new Image(
                    handle,
                    dimension,
                    pixelFormat,
                    width,
                    height,
                    depth,
                    arrayLayers,
                    mipLevels,
                    options,
                    mapped,

                    allocation,
                    cx,
                    local
            );
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private Image(
            VkImage handle,
            Dimension dimension,
            PixelFormat pixelFormat,
            int width,
            int height,
            int depth,
            int arrayLayers,
            int mipLevels,
            Options options,

            @Nullable BytePtr mapped,
            VmaAllocation allocation,
            RenderContext context,
            boolean local
    ) {
        this.handle = handle;
        this.dimension = dimension;
        this.pixelFormat = pixelFormat;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.arrayLayers = arrayLayers;
        this.mipLevels = mipLevels;
        this.mapped = mapped;
        this.options = options;

        IDisposeOnContext d = cx -> cx.vma.destroyImage(cx.vmaAllocator, handle, allocation);
        this.cleanable = context.registerCleanup(this, d, local);
    }

    private final Cleaner.Cleanable cleanable;
}
