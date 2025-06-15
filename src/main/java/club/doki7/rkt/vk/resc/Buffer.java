package club.doki7.rkt.vk.resc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.PointerPtr;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vma.bitmask.VmaAllocationCreateFlags;
import club.doki7.vma.datatype.VmaAllocationCreateInfo;
import club.doki7.vma.datatype.VmaAllocationInfo;
import club.doki7.vma.enumtype.VmaMemoryUsage;
import club.doki7.vma.handle.VmaAllocation;
import club.doki7.vulkan.bitmask.VkBufferUsageFlags;
import club.doki7.vulkan.datatype.VkBufferCreateInfo;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.enumtype.VkSharingMode;
import club.doki7.vulkan.handle.VkBuffer;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;

public final class Buffer implements AutoCloseable {
    public final VkBuffer handle;
    public final long size;
    public final boolean hostAccess;
    public final @Nullable BytePtr mapped;

    public static Buffer create(
            RenderContext cx,
            long size,
            boolean local,
            @EnumType(VkBufferUsageFlags.class) int usage,
            @EnumType(VmaAllocationCreateFlags.class) int allocationFlags,
            @EnumType(VkSharingMode.class) int sharingMode
    ) throws VulkanException {
        boolean hostAccess = (allocationFlags & VmaAllocationCreateFlags.HOST_ACCESS_RANDOM) != 0;
        boolean initiallyMapped = (allocationFlags & VmaAllocationCreateFlags.MAPPED) != 0;

        try (Arena arena = Arena.ofConfined()) {
            VkBufferCreateInfo createInfo = VkBufferCreateInfo.allocate(arena)
                    .size(size)
                    .usage(usage)
                    .sharingMode(sharingMode);
            VmaAllocationCreateInfo allocationCreateInfo = VmaAllocationCreateInfo.allocate(arena)
                    .usage(VmaMemoryUsage.AUTO)
                    .flags(allocationFlags);
            @Nullable VmaAllocationInfo allocationInfo = initiallyMapped
                    ? VmaAllocationInfo.allocate(arena)
                    : null;

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
            BytePtr mapped = allocationInfo != null
                    ? new BytePtr(allocationInfo.pMappedData().reinterpret(size))
                    : null;
            return new Buffer(handle, size, hostAccess, mapped, allocation, cx, local);
        }
    }

    public static Buffer createStaging(RenderContext cx, long size) throws VulkanException {
        return create(
                cx,
                size,
                true,
                VkBufferUsageFlags.TRANSFER_DST | VkBufferUsageFlags.TRANSFER_SRC,
                VmaAllocationCreateFlags.HOST_ACCESS_RANDOM | VmaAllocationCreateFlags.MAPPED,
                VkSharingMode.EXCLUSIVE
        );
    }

    public static Buffer createVertexBuffer(RenderContext cx, long size) throws VulkanException {
        return create(
                cx,
                size,
                false,
                VkBufferUsageFlags.VERTEX_BUFFER | VkBufferUsageFlags.TRANSFER_DST,
                0,
                VkSharingMode.EXCLUSIVE
        );
    }

    public static Buffer createIndexBuffer(RenderContext cx, long size) throws VulkanException {
        return create(
                cx,
                size,
                false,
                VkBufferUsageFlags.INDEX_BUFFER | VkBufferUsageFlags.TRANSFER_DST,
                0,
                VkSharingMode.EXCLUSIVE
        );
    }

    public static Buffer createUniformBuffer(RenderContext cx, long size) throws VulkanException {
        return create(
                cx,
                size,
                false,
                VkBufferUsageFlags.UNIFORM_BUFFER | VkBufferUsageFlags.TRANSFER_DST,
                VmaAllocationCreateFlags.HOST_ACCESS_RANDOM | VmaAllocationCreateFlags.MAPPED,
                VkSharingMode.EXCLUSIVE
        );
    }

    public static Buffer createShaderStorageBuffer(
            RenderContext cx,
            long size
    ) throws VulkanException {
        return create(
                cx,
                size,
                false,
                VkBufferUsageFlags.STORAGE_BUFFER | VkBufferUsageFlags.TRANSFER_DST,
                VmaAllocationCreateFlags.HOST_ACCESS_RANDOM | VmaAllocationCreateFlags.MAPPED,
                VkSharingMode.EXCLUSIVE
        );
    }

    public BytePtr map(RenderContext cx) throws VulkanException {
        if (!hostAccess) {
            throw new IllegalStateException("此缓冲不支持映射");
        }

        if (mapped != null) {
            return mapped;
        }

        try (Arena arena = Arena.ofConfined()) {
            PointerPtr ppData = PointerPtr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    cx.vma.mapMemory(cx.vmaAllocator, allocation, ppData);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法映射 Vulkan 缓冲区");
            }
            return new BytePtr(ppData.read().reinterpret(size));
        }
    }

    public void unmap(RenderContext cx) {
        if (!hostAccess) {
            throw new IllegalStateException("此缓冲不支持映射");
        }

        if (mapped != null) {
            return;
        }

        cx.vma.unmapMemory(cx.vmaAllocator, allocation);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private Buffer(
            VkBuffer handle,
            long size,
            boolean hostAccess,
            @Nullable BytePtr mapped,
            VmaAllocation allocation,
            RenderContext context,
            boolean local
    ) {
        this.handle = handle;
        this.size = size;
        this.hostAccess = hostAccess;
        this.mapped = mapped;
        this.allocation = allocation;
        IDisposeOnContext d = cx -> cx.vma.destroyBuffer(cx.vmaAllocator, handle, allocation);
        if (local) {
            this.cleanable = context.cleaner.register(this, () -> context.disposeImmediate(d));
        } else {
            this.cleanable = context.cleaner.register(this, () -> context.dispose(d));
        }
    }

    private final VmaAllocation allocation;
    private final Cleaner.Cleanable cleanable;
}
