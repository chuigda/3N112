package club.doki7.rkt.launch.nn;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.library.ILibraryLoader;
import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.rkt.exc.RenderException;
import club.doki7.rkt.shaderc.ShaderCompiler;
import club.doki7.rkt.vk.RenderConfig;
import club.doki7.rkt.vk.RenderContext;
import club.doki7.rkt.vk.common.QueueFamily;
import club.doki7.rkt.vk.resc.Buffer;
import club.doki7.rkt.vk.resc.Transmission;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercIncludeType;
import club.doki7.vulkan.command.VulkanLoader;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.logging.Logger;

public final class MLP {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tFT%1$tT] [%4$s] %3$s : %5$s%n");
    }

    public static void main(String[] args) {
        try (ISharedLibrary libVulkan = VulkanLoader.loadVulkanLibrary();
             ISharedLibrary libVMA = ILibraryLoader.platformLoader().loadLibrary("vma");
             ISharedLibrary libShaderc = ILibraryLoader.platformLoader().loadLibrary("shaderc_shared");
             Application app = new Application(libVulkan, libVMA, libShaderc)) {
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
        compiler.close();
    }

    Application(ISharedLibrary libVulkan, ISharedLibrary libVMA, ISharedLibrary libShaderc) throws RenderException {
        this.cx = RenderContext.createHeadless(libVulkan, libVMA, new RenderConfig());
        this.compiler = ShaderCompiler.create(new Shaderc(libShaderc), Application::rescDirResolve);
    }

    void applicationStart() throws RenderException, IOException {
        Buffer.OptionsInit weightOptionsInit = Buffer.OptionsInit.shaderStorageBufferPreset();
        weightOptionsInit.usage = Set.of(Buffer.Usage.TRANSFER_DST, Buffer.Usage.STORAGE_BUFFER);
        Buffer.Options weightOptions = weightOptionsInit.build();

        byte[] weightsDataL1 = Files.readAllBytes(Path.of("resc", "nn", "weights_L1_784x300.bin"));
        byte[] weightsDataL2 = Files.readAllBytes(Path.of("resc", "nn", "weights_L2_300x100.bin"));
        byte[] weightsDataL3 = Files.readAllBytes(Path.of("resc", "nn", "weights_L3_100x10.bin"));
        byte[] biasesDataL1 = Files.readAllBytes(Path.of("resc", "nn", "biases_L1_784x300.bin"));
        byte[] biasesDataL2 = Files.readAllBytes(Path.of("resc", "nn", "biases_L2_300x100.bin"));
        byte[] biasesDataL3 = Files.readAllBytes(Path.of("resc", "nn", "biases_L3_100x10.bin"));

        assert weightsDataL1.length == 784 * 300 * Float.BYTES : "Weights data size mismatch";
        assert weightsDataL2.length == 300 * 100 * Float.BYTES : "Weights data size mismatch";
        assert weightsDataL3.length == 100 * 10 * Float.BYTES : "Weights data size mismatch";
        assert biasesDataL1.length == 300 * Float.BYTES : "Biases data size mismatch";
        assert biasesDataL2.length == 100 * Float.BYTES : "Biases data size mismatch";
        assert biasesDataL3.length == 10 * Float.BYTES : "Biases data size mismatch";

        try (Buffer weightsL1 = Buffer.create(cx, 784 * 300 * Float.BYTES, true, weightOptions);
             Buffer weightsL2 = Buffer.create(cx, 300 * 100 * Float.BYTES, true, weightOptions);
             Buffer weightsL3 = Buffer.create(cx, 100 * 10 * Float.BYTES, true, weightOptions);
             Buffer biasesL1 = Buffer.create(cx, 300 * Float.BYTES, true, weightOptions);
             Buffer biasesL2 = Buffer.create(cx, 100 * Float.BYTES, true, weightOptions);
             Buffer biasesL3 = Buffer.create(cx, 10 * Float.BYTES, true, weightOptions)) {
            Transmission.uploadBuffer(cx, weightsL1, MemorySegment.ofArray(weightsDataL1), QueueFamily.COMPUTE);
            Transmission.uploadBuffer(cx, weightsL2, MemorySegment.ofArray(weightsDataL2), QueueFamily.COMPUTE);
            Transmission.uploadBuffer(cx, weightsL3, MemorySegment.ofArray(weightsDataL3), QueueFamily.COMPUTE);
            Transmission.uploadBuffer(cx, biasesL1, MemorySegment.ofArray(biasesDataL1), QueueFamily.COMPUTE);
            Transmission.uploadBuffer(cx, biasesL2, MemorySegment.ofArray(biasesDataL2), QueueFamily.COMPUTE);
            Transmission.uploadBuffer(cx, biasesL3, MemorySegment.ofArray(biasesDataL3), QueueFamily.COMPUTE);
        }
    }

    private void loadWeights() {
    }

    private final RenderContext cx;
    private final ShaderCompiler compiler;

    private Buffer[] weightsBuffer;
    private Buffer[] biasesBuffer;

    private static ShadercUtil.IncludeResult rescDirResolve(
            String requestedSource,
            @EnumType(ShadercIncludeType.class) int includeType,
            String requestingSource,
            long includeDepth
    ) throws IOException {
        Path path = Path.of("resc", "shader", requestedSource);
        String content = Files.readString(path);
        return new ShadercUtil.IncludeResult(path.toAbsolutePath().toString(), content);
    }

    private static final Logger logger = Logger.getLogger(Application.class.getName());
}
