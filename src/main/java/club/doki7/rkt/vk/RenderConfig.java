package club.doki7.rkt.vk;

import club.doki7.vulkan.Version;
import club.doki7.vulkan.datatype.VkPhysicalDeviceProperties;

public final class RenderConfig {
    @FunctionalInterface
    public interface PhysicalDeviceRanker {
        /// 给予物理设备一个评分，越大越好。负分表示该设备不合适。
        int rank(VkPhysicalDeviceProperties properties);

        /// 默认的物理设备评分器，总是返回 0。
        static PhysicalDeviceRanker dummy() {
            return ignored -> 0;
        }
    }

    /// 应用程序名称，作为提示信息传入
    /// {@link club.doki7.vulkan.datatype.VkApplicationInfo VkApplicationInfo}。
    public String appName = "Vulkan Application";
    /// 应用程序版本，作为提示信息传入
    /// {@link club.doki7.vulkan.datatype.VkApplicationInfo VkApplicationInfo}。
    public Version appVersion = new Version(1, 0, 0, 0);

    /// 物理设备选择器。渲染引擎会根据选择器给出的评分对设备进行排序，选择评分最高的设备。
    public PhysicalDeviceRanker physicalDeviceRanker = PhysicalDeviceRanker.dummy();
    /// 无论物理设备是否具有专用传输队列，都不使用专用传输队列上传数据。
    public boolean noTransferQueue = false;
    /// 无论物理设备是否具有专用计算队列，都不使用专用计算队列进行计算。
    public boolean noComputeQueue = false;

    /// 是否启用主机复制，这在特定情况下可以提高性能，但某些设备可能不支持。
    public boolean enableHostCopy = false;

    /// 是否启用各向异性过滤。
    public boolean enableAnisotropicFiltering = true;

    public enum VSync { OFF, PREFER_OFF, ON }

    /// 垂直同步：0 为总是关闭，1 为倾向于关闭，2 为总是开启。
    ///
    /// 技术上来说，这个值设为 1 时，渲染引擎会查询交换链是否支持
    /// {@link club.doki7.vulkan.enumtype.VkPresentModeKHR#MAILBOX VkPresentMode.MAILBOX}，
    /// 如果支持则使用该模式，否则回退到
    /// {@link club.doki7.vulkan.enumtype.VkPresentModeKHR#FIFO VkPresentMode.FIFO}。当这个值设为 0
    /// 时，如果 {@link club.doki7.vulkan.enumtype.VkPresentModeKHR#MAILBOX VkPresentMode.MAILBOX}
    /// 模式不可用，则渲染引擎会回退到
    /// {@link club.doki7.vulkan.enumtype.VkPresentModeKHR#IMMEDIATE VkPresentMode.IMMEDIATE}。当
    /// 这个值设为 2 时，渲染引擎总是会使用
    /// {@link club.doki7.vulkan.enumtype.VkPresentModeKHR#FIFO VkPresentMode.FIFO} 模式。
    ///
    /// @see <a href="https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html"><code>VkPresentModeKHR</code></a>
    public VSync vsync = VSync.PREFER_OFF;

    /// 允许同时渲染多少帧。
    public int maxFramesInFlight = 2;
}
