package club.doki7.rkt.vk.resc;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vma.bitmask.VmaAllocationCreateFlags;
import club.doki7.vma.datatype.VmaAllocationCreateInfo;
import club.doki7.vma.datatype.VmaAllocationInfo;
import club.doki7.vma.enumtype.VmaMemoryUsage;
import club.doki7.vma.handle.VmaAllocation;
import club.doki7.vulkan.bitmask.VkBufferUsageFlags;
import club.doki7.vulkan.bitmask.VkMemoryPropertyFlags;
import club.doki7.vulkan.datatype.VkBufferCreateInfo;
import club.doki7.vulkan.datatype.VkMappedMemoryRange;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.enumtype.VkSharingMode;
import club.doki7.vulkan.handle.VkBuffer;
import club.doki7.vulkan.handle.VkDeviceMemory;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Collection;
import java.util.Collections;
import java.util.Set;
import java.util.function.Consumer;

public final class Buffer implements AutoCloseable {
    public enum Usage {
        VERTEX_BUFFER(VkBufferUsageFlags.VERTEX_BUFFER),
        INDEX_BUFFER(VkBufferUsageFlags.INDEX_BUFFER),
        INDIRECT_BUFFER(VkBufferUsageFlags.INDIRECT_BUFFER),
        UNIFORM_BUFFER(VkBufferUsageFlags.UNIFORM_BUFFER),
        TRANSFER_SRC(VkBufferUsageFlags.TRANSFER_SRC),
        TRANSFER_DST(VkBufferUsageFlags.TRANSFER_DST),
        STORAGE_BUFFER(VkBufferUsageFlags.STORAGE_BUFFER);

        @Bitmask(VkBufferUsageFlags.class) final int flag;

        Usage(@Bitmask(VkBufferUsageFlags.class) int flag) {
            this.flag = flag;
        }
    }

    public static final class Options {
        public final Set<Usage> usage;
        public final boolean mapped;
        public final boolean coherent;
        public final boolean shared;

        final @Bitmask(VkBufferUsageFlags.class) int usageFlags;
        final @Bitmask(VmaAllocationCreateFlags.class) int allocationCreateFlags;
        final @Bitmask(VkMemoryPropertyFlags.class) int memoryPropertyFlags;
        final @EnumType(VkSharingMode.class) int sharingMode;

        Options(
                Set<Usage> usage,
                boolean mapped,
                boolean coherent,
                boolean shared
        ) {
            this.usage = Collections.unmodifiableSet(usage);
            this.mapped = mapped;
            this.coherent = coherent;
            this.shared = shared;

            @Bitmask(VkBufferUsageFlags.class) int usageFlags = 0;
            for (Usage u : usage) {
                usageFlags |= u.flag;
            }
            this.usageFlags = usageFlags;

            @Bitmask(VmaAllocationCreateFlags.class) int allocationCreateFlags = 0;
            if (mapped) {
                allocationCreateFlags |= VmaAllocationCreateFlags.MAPPED;
                allocationCreateFlags |= VmaAllocationCreateFlags.HOST_ACCESS_RANDOM;
            }
            this.allocationCreateFlags = allocationCreateFlags;

            @Bitmask(VkMemoryPropertyFlags.class) int memoryPropertyFlags = coherent
                    ? 0
                    : VkMemoryPropertyFlags.HOST_VISIBLE;
            this.memoryPropertyFlags = memoryPropertyFlags;

            this.sharingMode = shared
                    ? VkSharingMode.CONCURRENT
                    : VkSharingMode.EXCLUSIVE;
        }

        public static Options init(Consumer<OptionsInit> consumer) {
            OptionsInit ret = new OptionsInit();
            consumer.accept(ret);
            return ret.build();
        }
    }

    public static final class OptionsInit {
        public Set<Usage> usage;
        public boolean mapped;
        public boolean coherent;
        public boolean shared;

        public Options build() {
            if (coherent && !mapped) {
                throw new IllegalStateException("无效的参数组合：若指定了 coherent，则必须指定 mapped");
            }

            if (usage.isEmpty()) {
                throw new IllegalStateException("使用的缓冲区类型不能为空");
            }

            return new Options(usage, mapped, coherent, shared);
        }

        public OptionsInit() {
            this.usage = Collections.emptySet();
            this.mapped = false;
            this.coherent = false;
            this.shared = false;
        }

        public static OptionsInit vertexBufferPreset() {
            OptionsInit init = new OptionsInit();
            init.usage = Set.of(Usage.VERTEX_BUFFER, Usage.TRANSFER_DST);
            init.mapped = false;
            init.coherent = false;
            init.shared = false;
            return init;
        }

        public static OptionsInit indexBufferPreset() {
            OptionsInit init = new OptionsInit();
            init.usage = Set.of(Usage.INDEX_BUFFER, Usage.TRANSFER_DST);
            init.mapped = false;
            init.coherent = false;
            init.shared = false;
            return init;
        }

        private static OptionsInit uniformBufferPreset() {
            OptionsInit init = new OptionsInit();
            init.usage = Set.of(Usage.UNIFORM_BUFFER);
            init.mapped = true;
            init.coherent = false;
            init.shared = false;
            return init;
        }

        public static OptionsInit stagingBufferPreset() {
            OptionsInit init = new OptionsInit();
            init.usage = Set.of(Usage.TRANSFER_SRC, Usage.TRANSFER_DST);
            init.mapped = true;
            init.coherent = false;
            init.shared = false;
            return init;
        }

        public static OptionsInit shaderStorageBufferPreset() {
            OptionsInit init = new OptionsInit();
            init.usage = Set.of(Usage.STORAGE_BUFFER);
            init.mapped = false;
            init.coherent = false;
            init.shared = false;
            return init;
        }
    }

    public final VkBuffer handle;
    public final long size;
    public final Options options;

    public final VkDeviceMemory deviceMemory;
    public final MemorySegment mapped;

    public void invalidate(RenderContext cx) throws VulkanException {
        if (mappedMemoryRange == null) {
            throw new IllegalStateException("缓冲未被映射或者不需要失效");
        }

        @EnumType(VkResult.class) int result = cx.dCmd.invalidateMappedMemoryRanges(
                cx.device,
                1,
                mappedMemoryRange
        );
        if (result != VkResult.SUCCESS) {
            throw new VulkanException(result, "无法失效 Vulkan 缓冲区的映射内存范围");
        }
    }

    public void flush(RenderContext cx) throws VulkanException {
        if (mappedMemoryRange == null) {
            throw new IllegalStateException("缓冲未被映射或者不需要刷新");
        }

        @EnumType(VkResult.class) int result = cx.dCmd.flushMappedMemoryRanges(
                cx.device,
                1,
                mappedMemoryRange
        );
        if (result != VkResult.SUCCESS) {
            throw new VulkanException(result, "无法刷新 Vulkan 缓冲区的映射内存范围");
        }
    }

    public static void invalidate(
            RenderContext cx,
            Collection<Buffer> buffers
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkMappedMemoryRange.Ptr ranges = VkMappedMemoryRange.allocate(arena, buffers.size());
            long index = 0;
            for (Buffer buffer : buffers) {
                if (buffer.mappedMemoryRange == null) {
                    throw new IllegalStateException("缓冲 " + buffer.handle + " 未被映射或者不需要失效");
                }
                ranges.write(index, buffer.mappedMemoryRange);
                index += 1;
            }

            @EnumType(VkResult.class) int result = cx.dCmd.invalidateMappedMemoryRanges(
                    cx.device,
                    (int) ranges.size(),
                    ranges
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法失效 Vulkan 缓冲区的映射内存范围");
            }
        }
    }

    public static void flush(
            RenderContext cx,
            Collection<Buffer> buffers
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkMappedMemoryRange.Ptr ranges = VkMappedMemoryRange.allocate(arena, buffers.size());
            long index = 0;
            for (Buffer buffer : buffers) {
                if (buffer.mappedMemoryRange == null) {
                    throw new IllegalStateException("缓冲 " + buffer.handle + " 未被映射或者不需要刷新");
                }
                ranges.write(index, buffer.mappedMemoryRange);
                index += 1;
            }

            @EnumType(VkResult.class) int result = cx.dCmd.flushMappedMemoryRanges(
                    cx.device,
                    (int) ranges.size(),
                    ranges
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法刷新 Vulkan 缓冲区的映射内存范围");
            }
        }
    }

    public static Buffer create(
            RenderContext cx,
            long size,
            boolean local,
            Options options
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkBufferCreateInfo createInfo = VkBufferCreateInfo.allocate(arena)
                    .size(size)
                    .usage(options.usageFlags)
                    .sharingMode(options.sharingMode);
            VmaAllocationCreateInfo allocationCreateInfo = VmaAllocationCreateInfo.allocate(arena)
                    .usage(VmaMemoryUsage.AUTO)
                    .flags(options.allocationCreateFlags)
                    .requiredFlags(options.memoryPropertyFlags);
            VmaAllocationInfo allocationInfo = VmaAllocationInfo.allocate(arena);

            VkBuffer.Ptr pBuffer = VkBuffer.Ptr.allocate(arena);
            VmaAllocation.Ptr pAllocation = VmaAllocation.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = cx.vma.createBuffer(
                    cx.vmaAllocator,
                    createInfo,
                    allocationCreateInfo,
                    pBuffer,
                    pAllocation,
                    allocationInfo
            );
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 缓冲区");
            }

            VkBuffer handle = pBuffer.read();
            VmaAllocation allocation = pAllocation.read();
            VkDeviceMemory deviceMemory = allocationInfo.deviceMemory();
            MemorySegment mapped = options.mapped
                    ? allocationInfo.pMappedData().reinterpret(size)
                    : MemorySegment.NULL;
            return new Buffer(handle, size, options, deviceMemory, mapped, allocation, cx, local);
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private Buffer(
            VkBuffer handle,
            long size,
            Options options,
            VkDeviceMemory deviceMemory,
            MemorySegment mapped,

            VmaAllocation allocation,
            RenderContext context,
            boolean local
    ) {
        this.handle = handle;
        this.size = size;
        this.options = options;
        this.deviceMemory = deviceMemory;
        this.mapped = mapped;

        if (options.mapped && !options.coherent) {
            this.mappedMemoryRange = VkMappedMemoryRange.allocate(context.prefabArena)
                    .memory(deviceMemory)
                    .size(size)
                    .offset(0);
        } else {
            this.mappedMemoryRange = null;
        }

        IDisposeOnContext d = cx -> cx.vma.destroyBuffer(cx.vmaAllocator, handle, allocation);
        this.cleanable = context.registerCleanup(this, d, local);
    }

    final @Nullable VkMappedMemoryRange mappedMemoryRange;
    private final Cleaner.Cleanable cleanable;
}
