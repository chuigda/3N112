package club.doki7.sputnik.vk.init;

import club.doki7.sputnik.exc.RenderException;
import club.doki7.sputnik.exc.VulkanException;
import club.doki7.sputnik.util.Assertion;
import club.doki7.sputnik.vk.RenderConfig;
import club.doki7.sputnik.vk.RenderContext;
import club.doki7.ffm.Loader;
import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.*;
import club.doki7.glfw.GLFW;
import club.doki7.glfw.handle.GLFWwindow;
import club.doki7.vma.VMA;
import club.doki7.vma.VMAJavaTraceUtil;
import club.doki7.vma.VMAUtil;
import club.doki7.vma.datatype.VmaAllocatorCreateInfo;
import club.doki7.vma.datatype.VmaVulkanFunctions;
import club.doki7.vma.handle.VmaAllocator;
import club.doki7.vulkan.Version;
import club.doki7.vulkan.VkConstants;
import club.doki7.vulkan.bitmask.*;
import club.doki7.vulkan.command.*;
import club.doki7.vulkan.datatype.*;
import club.doki7.vulkan.enumtype.*;
import club.doki7.vulkan.handle.*;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.util.Objects;
import java.util.logging.Logger;

public final class ContextInit {
    private final Arena prefabArena = Arena.ofAuto();

    private final GLFW glfw;
    private final GLFWwindow window;

    private final RenderConfig config;

    private VkStaticCommands sCmd;
    private VkEntryCommands eCmd;

    private boolean enableValidationLayers;

    private VkInstance instance;
    private VkInstanceCommands iCmd;
    private @Nullable VkDebugUtilsMessengerEXT debugMessenger;
    private VkSurfaceKHR surface;

    private VkPhysicalDevice physicalDevice;
    private int graphicsQueueFamilyIndex;
    private int presentQueueFamilyIndex;
    private int dedicatedTransferQueueFamilyIndex;
    private int dedicatedComputeQueueFamilyIndex;

    private VkDevice device;
    private VkDeviceCommands dCmd;
    private VkQueue graphicsQueue;
    private VkQueue presentQueue;
    private @Nullable VkQueue dedicatedTransferQueue;
    private @Nullable VkQueue dedicatedComputeQueue;

    private VMA vma;
    private VmaAllocator vmaAllocator;

    public ContextInit(
            GLFW glfw,
            GLFWwindow window,
            RenderConfig config
    ) {
        this.glfw = glfw;
        this.window = window;
        this.config = config;
    }

    public RenderContext init() throws RenderException {
        sCmd = VulkanLoader.loadStaticCommands();
        eCmd = VulkanLoader.loadEntryCommands(sCmd);

        try {
            createInstance();
            setupDebugMessenger();
            createSurface();
            pickPhysicalDevice();
            findQueueFamilyIndices();
            createLogicalDevice();
            createVMA();
        } catch (Throwable e) {
            cleanup();
            throw e;
        }

        return new RenderContext(
                prefabArena,
                config,

                sCmd,
                eCmd,
                iCmd,
                dCmd,
                vma,

                physicalDevice,
                graphicsQueueFamilyIndex,
                presentQueueFamilyIndex,
                dedicatedTransferQueueFamilyIndex,
                dedicatedComputeQueueFamilyIndex,

                instance,
                debugMessenger,
                surface,

                device,
                graphicsQueue,
                presentQueue,
                dedicatedTransferQueue,
                dedicatedComputeQueue,

                vmaAllocator
        );
    }

    private void createInstance() throws RenderException {
        enableValidationLayers = Assertion.assertionEnabled;
        if (enableValidationLayers) {
            logger.info("检测到启用了 Java 断言，将尝试启用 Vulkan 校验层");
        }

        boolean validationLayersSupported = checkValidationLayerSupport();
        if (enableValidationLayers && !validationLayersSupported) {
            logger.warning("检测到启用了 Java 断言，但 Vulkan 校验层 " + VALIDATION_LAYER_NAME + " 不可用，将禁用校验层");
            enableValidationLayers = false;
        } else {
            logger.info("Vulkan 校验层可用，将启用校验层");
        }

        try (Arena arena = Arena.ofConfined()) {
            VkApplicationInfo appInfo = VkApplicationInfo.allocate(arena)
                    .pApplicationName(BytePtr.allocateString(arena, config.appName))
                    .applicationVersion(config.appVersion.encode())
                    .pEngineName(BytePtr.allocateString(arena, "CG-112"))
                    .engineVersion(new Version(0x07, 0x21, 0x0D, 0x00).encode())
                    .apiVersion(Version.VK_API_VERSION_1_0.encode());

            IntPtr pGLFWExtensionCount = IntPtr.allocate(arena);
            PointerPtr glfwExtensions = glfw.getRequiredInstanceExtensions(pGLFWExtensionCount);
            if (glfwExtensions == null) {
                throw new RenderException("无法获取 GLFW 所需的 Vulkan 实例扩展");
            }
            int glfwExtensionCount = pGLFWExtensionCount.read();
            glfwExtensions = glfwExtensions.reinterpret(glfwExtensionCount);

            PointerPtr extensions;
            if (enableValidationLayers) {
                extensions = PointerPtr.allocate(arena, glfwExtensionCount + 2);
            } else {
                extensions = PointerPtr.allocate(arena, glfwExtensionCount + 1);
            }

            for (int i = 0; i < glfwExtensionCount; i++) {
                extensions.write(i, glfwExtensions.read(i));
            }
            extensions.write(
                    glfwExtensionCount,
                    BytePtr.allocateString(
                            arena,
                            VkConstants.KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
                    )
            );
            if (enableValidationLayers) {
                extensions.write(
                        glfwExtensionCount + 1,
                        BytePtr.allocateString(arena, VkConstants.EXT_DEBUG_UTILS_EXTENSION_NAME)
                );
            }

            VkInstanceCreateInfo instanceCreateInfo = VkInstanceCreateInfo.allocate(arena)
                    .pApplicationInfo(appInfo)
                    .enabledExtensionCount((int) extensions.size())
                    .ppEnabledExtensionNames(extensions);

            if (enableValidationLayers) {
                PointerPtr ppEnabledLayerNames = PointerPtr.allocateV(
                        arena,
                        BytePtr.allocateString(arena, VALIDATION_LAYER_NAME)
                );
                instanceCreateInfo.enabledLayerCount(1).ppEnabledLayerNames(ppEnabledLayerNames);

                VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
                        VkDebugUtilsMessengerCreateInfoEXT.allocate(arena);
                populateDebugMessengerCreateInfo(debugCreateInfo);
                instanceCreateInfo.pNext(debugCreateInfo);
            }

            VkInstance.Ptr pInstance = VkInstance.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    eCmd.createInstance(instanceCreateInfo, null, pInstance);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 实例");
            }

            instance = Objects.requireNonNull(pInstance.read());
            iCmd = VulkanLoader.loadInstanceCommands(instance, sCmd);
        }
    }

    private void setupDebugMessenger() {
        if (!enableValidationLayers) {
            debugMessenger = null;
            return;
        }

        try (Arena arena = Arena.ofConfined()) {
            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = VkDebugUtilsMessengerCreateInfoEXT.allocate(arena);
            populateDebugMessengerCreateInfo(debugCreateInfo);

            VkDebugUtilsMessengerEXT.Ptr pDebugMessenger = VkDebugUtilsMessengerEXT.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = iCmd.createDebugUtilsMessengerEXT(
                    instance,
                    debugCreateInfo,
                    null,
                    pDebugMessenger
            );
            if (result != VkResult.SUCCESS) {
                logger.severe("无法创建 Vulkan 调试信使, 错误代码: " + VkResult.explain(result));
                logger.warning("程序将会继续运行, 但校验层调试信息可能无法输出");
                debugMessenger = null;
            } else {
                debugMessenger = Objects.requireNonNull(pDebugMessenger.read());
            }
        }
    }

    private void createSurface() throws RenderException {
        try (Arena arena = Arena.ofConfined()) {
            VkSurfaceKHR.Ptr pSurface = VkSurfaceKHR.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result = glfw.createWindowSurface(instance, window, null, pSurface);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 窗口表面");
            }
            surface = Objects.requireNonNull(pSurface.read());
        }
    }

    private void pickPhysicalDevice() throws RenderException {
        try (Arena arena = Arena.ofConfined()) {
            logger.info("正在选取 Vulkan 物理设备");
            IntPtr pDeviceCount = IntPtr.allocate(arena);
            @EnumType(VkResult.class) int result = iCmd.enumeratePhysicalDevices(instance, pDeviceCount, null);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 物理设备列表");
            }

            int deviceCount = pDeviceCount.read();
            if (deviceCount == 0) {
                throw new RenderException("未找到任何 Vulkan 物理设备");
            }

            VkPhysicalDevice.Ptr pDevices = VkPhysicalDevice.Ptr.allocate(arena, deviceCount);
            result = iCmd.enumeratePhysicalDevices(instance, pDeviceCount, pDevices);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法获取 Vulkan 物理设备列表");
            }

            VkPhysicalDeviceProperties.Ptr deviceProperties = VkPhysicalDeviceProperties.allocate(arena, deviceCount);
            for (int i = 0; i < deviceCount; i++) {
                VkPhysicalDevice device = Objects.requireNonNull(pDevices.read(i));
                VkPhysicalDeviceProperties deviceProperty = deviceProperties.at(i);
                iCmd.getPhysicalDeviceProperties(device, deviceProperty);
                Version decodedVersion = Version.decode(deviceProperty.apiVersion());
                logger.info(String.format(
                        "发现 Vulkan 物理设备, ID: %s, 名称: %s, 类型: %s, API 版本: %d.%d.%d",
                        Integer.toUnsignedString(deviceProperty.deviceID()),
                        deviceProperty.deviceName().readString(),
                        VkPhysicalDeviceType.explain(deviceProperty.deviceType()),
                        decodedVersion.major(),
                        decodedVersion.minor(),
                        decodedVersion.patch()
                ));
            }

            physicalDevice = null;
            int bestRank = Integer.MIN_VALUE;
            for (int i = 0; i < deviceCount; i++) {
                VkPhysicalDeviceProperties deviceProperty = deviceProperties.at(i);
                int rank = config.physicalDeviceRanker.rank(deviceProperty);
                if (rank >= 0 && rank > bestRank) {
                    bestRank = rank;
                    physicalDevice = pDevices.read(i);
                }
            }

            if (physicalDevice == null) {
                throw new RenderException("未找到指定的 Vulkan 物理设备");
            }
        }
    }

    private void findQueueFamilyIndices() throws RenderException {
        graphicsQueueFamilyIndex = -1;
        presentQueueFamilyIndex = -1;
        dedicatedTransferQueueFamilyIndex = -1;
        dedicatedComputeQueueFamilyIndex = -1;

        try (Arena arena = Arena.ofConfined()) {
            IntPtr pQueueFamilyPropertyCount = IntPtr.allocate(arena);
            iCmd.getPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, null);
            int queueFamilyPropertyCount = pQueueFamilyPropertyCount.read();
            VkQueueFamilyProperties.Ptr queueFamilyProperties =
                    VkQueueFamilyProperties.allocate(arena, queueFamilyPropertyCount);
            iCmd.getPhysicalDeviceQueueFamilyProperties(
                    physicalDevice,
                    pQueueFamilyPropertyCount,
                    queueFamilyProperties
            );

            for (int i = 0; i < queueFamilyPropertyCount; i++) {
                VkQueueFamilyProperties queueFamilyProperty = queueFamilyProperties.at(i);
                @EnumType(VkQueueFlags.class) int queueFlags = queueFamilyProperty.queueFlags();
                logger.fine("正在检查队列 " + i + ", 支持操作: " + VkQueueFlags.explain(queueFlags));

                if ((queueFlags & VkQueueFlags.GRAPHICS) != 0 && graphicsQueueFamilyIndex == -1) {
                    logger.info(
                            "找到支持图形渲染的队列族: " + i +
                            ", 队列数量: " + queueFamilyProperty.queueCount() +
                            ", 支持的操作: " + VkQueueFlags.explain(queueFlags)
                    );
                    graphicsQueueFamilyIndex = i;
                }

                IntPtr pSupportsPresent = IntPtr.allocate(arena);
                iCmd.getPhysicalDeviceSurfaceSupportKHR(
                        physicalDevice,
                        1,
                        surface,
                        pSupportsPresent
                );
                int supportsPresent = pSupportsPresent.read();
                if (supportsPresent == VkConstants.TRUE && presentQueueFamilyIndex == -1) {
                    logger.info("找到支持窗口呈现的队列族: " + i);
                    presentQueueFamilyIndex = i;
                }

                if (!config.noTransferQueue) {
                    @EnumType(VkQueueFlags.class) int prohibitedFlags =
                            VkQueueFlags.GRAPHICS | VkQueueFlags.COMPUTE;
                    if ((queueFlags & VkQueueFlags.TRANSFER) != 0
                        && (queueFlags & prohibitedFlags) == 0
                        && dedicatedTransferQueueFamilyIndex == -1) {
                        logger.info(
                                "找到专用传输队列族: " + i +
                                ", 队列数量: " + queueFamilyProperty.queueCount() +
                                ", 支持的操作: " + VkQueueFlags.explain(queueFlags)
                        );
                        dedicatedTransferQueueFamilyIndex = i;
                    }
                }

                if (!config.noComputeQueue) {
                    @EnumType(VkQueueFlags.class) int prohibitedFlags = VkQueueFlags.GRAPHICS;
                    if ((queueFlags & VkQueueFlags.COMPUTE) != 0
                        && (queueFlags & prohibitedFlags) == 0
                        && dedicatedComputeQueueFamilyIndex == -1) {
                        logger.info(
                                "找到专用计算队列族: " + i +
                                ", 队列数量: " + queueFamilyProperty.queueCount() +
                                ", 支持的操作: " + VkQueueFlags.explain(queueFlags)
                        );
                        dedicatedComputeQueueFamilyIndex = i;
                    }
                }
            }

            if (graphicsQueueFamilyIndex == -1) {
                throw new RenderException("未找到支持图形渲染的 Vulkan 队列族");
            }
            if (presentQueueFamilyIndex == -1) {
                throw new RenderException("未找到支持窗口呈现的 Vulkan 队列族");
            }
            if (dedicatedTransferQueueFamilyIndex == -1) {
                logger.info("专用传输队列族未找到或被手动禁用, 渲染器将不会使用专用传输队列传输数据");
            }
            if (dedicatedComputeQueueFamilyIndex == -1) {
                logger.info("专用计算队列族未找到或被手动禁用, 渲染器将不会使用专用计算队列进行计算");
            }
        }
    }

    private void createLogicalDevice() throws RenderException {
        try (Arena arena = Arena.ofConfined()) {
            VkPhysicalDeviceFeatures deviceFeatures = VkPhysicalDeviceFeatures.allocate(arena);
            deviceFeatures.sampleRateShading(VkConstants.TRUE);
            if (config.enableAnisotropicFiltering) {
                deviceFeatures.samplerAnisotropy(VkConstants.TRUE);
            }

            FloatPtr pQueuePriorities = FloatPtr.allocateV(arena, 1.0f);

            int queueCreateInfoCount = graphicsQueueFamilyIndex != presentQueueFamilyIndex ? 2 : 1;
            if (dedicatedTransferQueueFamilyIndex != -1) {
                queueCreateInfoCount++;
            }
            if (dedicatedComputeQueueFamilyIndex != -1) {
                queueCreateInfoCount++;
            }

            VkDeviceQueueCreateInfo.Ptr queueCreateInfos = VkDeviceQueueCreateInfo.allocate(arena, queueCreateInfoCount);
            int nthCreateInfo = 0;
            queueCreateInfos.at(nthCreateInfo)
                    .queueCount(1)
                    .queueFamilyIndex(graphicsQueueFamilyIndex)
                    .pQueuePriorities(pQueuePriorities);
            nthCreateInfo += 1;
            if (graphicsQueueFamilyIndex != presentQueueFamilyIndex) {
                queueCreateInfos.at(nthCreateInfo)
                        .queueCount(1)
                        .queueFamilyIndex(presentQueueFamilyIndex)
                        .pQueuePriorities(pQueuePriorities);
                nthCreateInfo += 1;
            }
            if (dedicatedTransferQueueFamilyIndex != -1) {
                queueCreateInfos.at(nthCreateInfo)
                        .queueCount(1)
                        .queueFamilyIndex(dedicatedTransferQueueFamilyIndex)
                        .pQueuePriorities(pQueuePriorities);
                nthCreateInfo += 1;
            }
            if (dedicatedComputeQueueFamilyIndex != -1) {
                queueCreateInfos.at(nthCreateInfo)
                        .queueCount(1)
                        .queueFamilyIndex(dedicatedComputeQueueFamilyIndex)
                        .pQueuePriorities(pQueuePriorities);
            }

            PointerPtr ppDeviceExtensions;
            if (config.enableHostCopy) {
                ppDeviceExtensions = PointerPtr.allocate(arena, 1 + 5 + 3);
            } else {
                ppDeviceExtensions = PointerPtr.allocate(arena, 1 + 5);
            }

            ppDeviceExtensions.write(0, BytePtr.allocateString(arena, VkConstants.KHR_SWAPCHAIN_EXTENSION_NAME));
            ppDeviceExtensions.write(1, BytePtr.allocateString(arena, VkConstants.KHR_DYNAMIC_RENDERING_EXTENSION_NAME));
            ppDeviceExtensions.write(2, BytePtr.allocateString(arena, VkConstants.KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME));
            ppDeviceExtensions.write(3, BytePtr.allocateString(arena, VkConstants.KHR_CREATE_RENDERPASS_2_EXTENSION_NAME));
            ppDeviceExtensions.write(4, BytePtr.allocateString(arena, VkConstants.KHR_MULTIVIEW_EXTENSION_NAME));
            ppDeviceExtensions.write(5, BytePtr.allocateString(arena, VkConstants.KHR_MAINTENANCE_2_EXTENSION_NAME));

            if (config.enableHostCopy) {
                ppDeviceExtensions.write(6, BytePtr.allocateString(arena, VkConstants.EXT_HOST_IMAGE_COPY_EXTENSION_NAME));
                ppDeviceExtensions.write(7, BytePtr.allocateString(arena, VkConstants.KHR_COPY_COMMANDS_2_EXTENSION_NAME));
                ppDeviceExtensions.write(8, BytePtr.allocateString(arena, VkConstants.KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME));
            }

            VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures =
                    VkPhysicalDeviceDynamicRenderingFeatures.allocate(arena)
                            .dynamicRendering(VkConstants.TRUE);

            VkDeviceCreateInfo deviceCreateInfo = VkDeviceCreateInfo.allocate(arena)
                    .pEnabledFeatures(deviceFeatures)
                    .queueCreateInfoCount(queueCreateInfoCount)
                    .pQueueCreateInfos(queueCreateInfos)
                    .enabledExtensionCount((int) ppDeviceExtensions.size())
                    .ppEnabledExtensionNames(ppDeviceExtensions)
                    .pNext(dynamicRenderingFeatures);
            if (enableValidationLayers) {
                PointerPtr ppEnabledLayerNames = PointerPtr.allocate(arena);
                ppEnabledLayerNames.write(BytePtr.allocateString(arena, VALIDATION_LAYER_NAME));
                deviceCreateInfo.enabledLayerCount(1).ppEnabledLayerNames(ppEnabledLayerNames);
            }

            VkDevice.Ptr pDevice = VkDevice.Ptr.allocate(arena);
            @EnumType(VkResult.class) int result =
                    iCmd.createDevice(physicalDevice, deviceCreateInfo, null, pDevice);
            if (result != VkResult.SUCCESS) {
                throw new VulkanException(result, "无法创建 Vulkan 逻辑设备");
            }
            device = Objects.requireNonNull(pDevice.read());
            dCmd = VulkanLoader.loadDeviceCommands(device, sCmd);

            VkQueue.Ptr pQueue = VkQueue.Ptr.allocate(arena);
            dCmd.getDeviceQueue(device, graphicsQueueFamilyIndex, 0, pQueue);
            graphicsQueue = Objects.requireNonNull(pQueue.read());
            dCmd.getDeviceQueue(device, presentQueueFamilyIndex, 0, pQueue);
            presentQueue = Objects.requireNonNull(pQueue.read());

            if (dedicatedTransferQueueFamilyIndex != -1) {
                dCmd.getDeviceQueue(device, dedicatedTransferQueueFamilyIndex, 0, pQueue);
                dedicatedTransferQueue = Objects.requireNonNull(pQueue.read());
            } else {
                dedicatedTransferQueue = null;
            }
            if (dedicatedComputeQueueFamilyIndex != -1) {
                dCmd.getDeviceQueue(device, dedicatedComputeQueueFamilyIndex, 0, pQueue);
                dedicatedComputeQueue = Objects.requireNonNull(pQueue.read());
            } else {
                dedicatedComputeQueue = null;
            }
        }
    }

    private void createVMA() {
        vma = new VMA(Loader::loadFunctionOrNull);
        VMAJavaTraceUtil.enableJavaTraceForVMA();

        try (var arena = Arena.ofConfined()) {
            var vmaVulkanFunctions = VmaVulkanFunctions.allocate(arena);
            VMAUtil.fillVulkanFunctions(vmaVulkanFunctions, sCmd, eCmd, iCmd, dCmd);

            var vmaCreateInfo = VmaAllocatorCreateInfo.allocate(arena)
                    .instance(instance)
                    .physicalDevice(physicalDevice)
                    .device(device)
                    .pVulkanFunctions(vmaVulkanFunctions)
                    .vulkanApiVersion(Version.VK_API_VERSION_1_0.encode());

            var pVmaAllocator = VmaAllocator.Ptr.allocate(arena);
            var result = vma.createAllocator(vmaCreateInfo, pVmaAllocator);
            if (result != VkResult.SUCCESS) {
                throw new RuntimeException("Failed to create VMA allocator, vulkan error code: " + VkResult.explain(result));
            }

            vmaAllocator = Objects.requireNonNull(pVmaAllocator.read());
        }
    }

    private void cleanup() {
        if (vmaAllocator != null) {
            vma.destroyAllocator(vmaAllocator);
        }

        if (vmaAllocator != null) {
            vma.destroyAllocator(vmaAllocator);
        }

        if (device != null) {
            dCmd.destroyDevice(device, null);
        }

        if (surface != null) {
            iCmd.destroySurfaceKHR(instance, surface, null);
        }

        if (debugMessenger != null) {
            iCmd.destroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
        }
        if (instance != null) {
            iCmd.destroyInstance(instance, null);
        }
    }

    private boolean checkValidationLayerSupport() {
        try (Arena arena = Arena.ofConfined()) {
            IntPtr pLayerCount = IntPtr.allocate(arena);
            @EnumType(VkResult.class) int result = eCmd.enumerateInstanceLayerProperties(pLayerCount, null);
            if (result != VkResult.SUCCESS) {
                logger.warning("无法获取 Vulkan 实例层属性, 错误代码: " + VkResult.explain(result));
                return false;
            }

            int layerCount = pLayerCount.read();
            if (layerCount == 0) {
                return false;
            }

            VkLayerProperties.Ptr availableLayerProperties = VkLayerProperties.allocate(arena, layerCount);
            result = eCmd.enumerateInstanceLayerProperties(pLayerCount, availableLayerProperties);
            if (result != VkResult.SUCCESS) {
                logger.warning("无法获取 Vulkan 实例层属性, 错误代码: " + VkResult.explain(result));
                return false;
            }

            for (VkLayerProperties layerProperties : availableLayerProperties) {
                if (VALIDATION_LAYER_NAME.equals(layerProperties.layerName().readString())) {
                    logger.info("找到 Vulkan 校验层: " + VALIDATION_LAYER_NAME);
                    return true;
                }
            }
            return false;
        }
    }

    private static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo) {
        debugUtilsMessengerCreateInfo.messageSeverity(
                VkDebugUtilsMessageSeverityFlagsEXT.VERBOSE
                | VkDebugUtilsMessageSeverityFlagsEXT.WARNING
                | VkDebugUtilsMessageSeverityFlagsEXT.ERROR
        ).messageType(
                VkDebugUtilsMessageTypeFlagsEXT.GENERAL
                | VkDebugUtilsMessageTypeFlagsEXT.VALIDATION
                | VkDebugUtilsMessageTypeFlagsEXT.PERFORMANCE
        ).pfnUserCallback(DebugCallback.UPCALL_debugCallback);
    }

    private static final String VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";
    private static final Logger logger = Logger.getLogger(ContextInit.class.getName());
}
