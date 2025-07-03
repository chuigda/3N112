package club.doki7.rkt.drv;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.vulkan.command.VulkanLoader;

public final class MainHeadless {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] args) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma")) {
            applicationStart(libVulkan, libVMA);
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }

    private static void applicationStart(
            ISharedLibrary libVulkan,
            ISharedLibrary libVMA
    ) throws RenderException {
        RenderConfig config = new RenderConfig();
        RenderContext cx = RenderContext.createHeadless(libVulkan, libVMA, config);

        cx.close();
    }
}
