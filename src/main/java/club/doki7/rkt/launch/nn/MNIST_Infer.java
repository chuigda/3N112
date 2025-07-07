package club.doki7.rkt.launch.nn;

import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.vulkan.command.VulkanLoader;

import java.io.IOException;
import java.util.List;
import java.util.logging.Logger;

public final class MNIST_Infer {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] args) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             Application app = new Application(libVulkan, libVMA)) {
            app.applicationStart();
        } catch (Throwable e) {
            e.printStackTrace(System.err);
        }
    }
}

final class Application implements AutoCloseable {
    @Override
    public void close() {
        cx.close();
    }

    Application(ISharedLibrary libVulkan, ISharedLibrary libVMA) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
    }

    void applicationStart() throws RenderException, IOException {
        try (MLPFactory factory = new MLPFactory(cx)) {
            MLP model = factory.createInfer(new MLPOptions(
                    28 * 28,
                    List.of(
                            new MLPOptions.Layer(300, Activation.SIGMOID),
                            new MLPOptions.Layer(100, Activation.SIGMOID),
                            new MLPOptions.Layer(10, Activation.LINEAR)
                    ),
                    64,
                    true
            ));
            System.out.println("model successfully created");
        }
    }

    private void loadWeights() {
    }

    private final RenderContext cx;

    private Buffer[] weightsBuffer;
    private Buffer[] biasesBuffer;

    private static final Logger logger = Logger.getLogger(Application.class.getName());
}
