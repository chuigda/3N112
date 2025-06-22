package club.doki7.rkt.vk.cmd;

import club.doki7.ffm.annotation.Bitmask;
import club.doki7.rkt.exc.VulkanException;
import club.doki7.rkt.vk.IDisposeOnContext;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.annotation.Unsafe;
import club.doki7.vulkan.bitmask.VkCommandPoolCreateFlags;
import club.doki7.vulkan.datatype.VkCommandBufferAllocateInfo;
import club.doki7.vulkan.datatype.VkCommandPoolCreateInfo;
import club.doki7.vulkan.enumtype.VkCommandBufferLevel;
import club.doki7.vulkan.enumtype.VkResult;
import club.doki7.vulkan.handle.VkCommandBuffer;
import club.doki7.vulkan.handle.VkCommandPool;

import java.lang.foreign.Arena;
import java.lang.ref.Cleaner;
import java.util.Objects;

public final class CommandPool implements AutoCloseable {
    public final VkCommandPool handle;

    public static CommandPool create(
            RenderContext cx,
            @Bitmask(VkCommandPoolCreateFlags.class) int flags,
            int queueFamilyIndex
    ) throws VulkanException {
        return create(cx, queueFamilyIndex, flags, false);
    }

    @Unsafe
    public static CommandPool createLocal(
            RenderContext cx,
            @Bitmask(VkCommandPoolCreateFlags.class) int flags,
            int queueFamilyIndex
    ) throws VulkanException {
        return create(cx, queueFamilyIndex, flags, true);
    }

    public CommandBuffer allocCmdBuf(
            RenderContext cx,
            @EnumType(VkCommandBufferLevel.class) int level
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkCommandBuffer.Ptr pCommandBuffer = VkCommandBuffer.Ptr.allocate(arena);
            VkCommandBufferAllocateInfo allocateInfo = VkCommandBufferAllocateInfo.allocate(arena)
                    .commandPool(handle)
                    .level(level)
                    .commandBufferCount(1);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.allocateCommandBuffers(cx.device, allocateInfo, pCommandBuffer);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法分配 Vulkan 命令缓冲区");
            }

            VkCommandBuffer commandBuffer = Objects.requireNonNull(pCommandBuffer.read());
            boolean canReset = (flags & VkCommandPoolCreateFlags.RESET_COMMAND_BUFFER) != 0;
            return new CommandBuffer(commandBuffer, canReset, this);
        }
    }

    public CommandBuffer[] allocCmdBufN(
            RenderContext cx,
            @EnumType(VkCommandBufferLevel.class) int level,
            int count
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkCommandBuffer.Ptr pCommandBuffers = VkCommandBuffer.Ptr.allocate(arena, count);
            VkCommandBufferAllocateInfo allocateInfo = VkCommandBufferAllocateInfo.allocate(arena)
                    .commandPool(handle)
                    .level(level)
                    .commandBufferCount(count);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.allocateCommandBuffers(cx.device, allocateInfo, pCommandBuffers);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法分配 Vulkan 命令缓冲区");
            }

            CommandBuffer[] buffers = new CommandBuffer[count];
            boolean canReset = (flags & VkCommandPoolCreateFlags.RESET_COMMAND_BUFFER) != 0;
            for (int i = 0; i < count; i++) {
                VkCommandBuffer vkCommandBuffer = pCommandBuffers.read(i);
                buffers[i] = new CommandBuffer(vkCommandBuffer, canReset, this);
            }
            return buffers;
        }
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private static CommandPool create(
            RenderContext cx,
            int queueFamilyIndex,
            @Bitmask(VkCommandPoolCreateFlags.class) int flags,
            boolean local
    ) throws VulkanException {
        try (Arena arena = Arena.ofConfined()) {
            VkCommandPoolCreateInfo createInfo = VkCommandPoolCreateInfo.allocate(arena)
                    .flags(flags)
                    .queueFamilyIndex(queueFamilyIndex);
            VkCommandPool.Ptr pCommandPool = VkCommandPool.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    cx.dCmd.createCommandPool(cx.device, createInfo, null, pCommandPool);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 命令池");
            }
            return new CommandPool(pCommandPool.read(), flags, cx, local);
        }
    }

    private CommandPool(
            VkCommandPool handle,
            @Bitmask(VkCommandPoolCreateFlags.class) int flags,
            RenderContext context,
            boolean local
    ) {
        this.handle = handle;
        this.flags = flags;

        IDisposeOnContext d = cx -> cx.dCmd.destroyCommandPool(cx.device, handle, null);
        this.cleanable = context.registerCleanup(this, d, local);
    }

    private final @Bitmask(VkCommandPoolCreateFlags.class) int flags;
    private final Cleaner.Cleanable cleanable;
}
