package club.doki7.rkt.shaderc;

import club.doki7.ffm.annotation.EnumType;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.rkt.exc.ShaderCompileException;
import club.doki7.shaderc.Shaderc;
import club.doki7.shaderc.ShadercUtil;
import club.doki7.shaderc.enumtype.ShadercShaderKind;
import club.doki7.shaderc.handle.ShadercCompilationResult;
import club.doki7.shaderc.handle.ShadercCompileOptions;
import club.doki7.shaderc.handle.ShadercCompiler;
import org.jetbrains.annotations.Nullable;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.Objects;

public final class ShaderCompiler implements AutoCloseable {
    public String compileIntoAssembly(
            String fileName,
            String sourceCode,
            @EnumType(ShadercShaderKind.class) int shaderKind
    ) throws ShaderCompileException {
        @Nullable ShadercCompilationResult result = null;
        try (Arena arena = Arena.ofConfined()) {
            BytePtr pFileName = BytePtr.allocateString(arena, fileName);
            BytePtr pSourceCode = BytePtr.allocateString(arena, sourceCode);

            result = Objects.requireNonNull(shaderc.compileIntoSPVAssembly(
                    compiler,
                    pSourceCode,
                    pSourceCode.size() - 1,
                    shaderKind,
                    pFileName,
                    entryPoint,
                    options
            ));

            long numErrors = shaderc.resultGetNumErrors(result);
            if (numErrors != 0) {
                String errorMessage = Objects.requireNonNull(shaderc.resultGetErrorMessage(result))
                        .readString();
                throw new ShaderCompileException(errorMessage);
            }

            return Objects.requireNonNull(shaderc.resultGetBytes(result)).readString();
        } finally {
            if (result != null) {
                shaderc.resultRelease(result);
            }
        }
    }

    public BytePtr compileIntoSPV(
            Arena resultArena,
            String fileName,
            String sourceCode,
            @EnumType(ShadercShaderKind.class) int shaderKind
    ) throws ShaderCompileException {
        ShadercCompilationResult result = null;
        try (Arena arena = Arena.ofConfined()) {
            BytePtr pFileName = BytePtr.allocateString(arena, fileName);
            BytePtr pSourceCode = BytePtr.allocateString(arena, sourceCode);

            result = Objects.requireNonNull(shaderc.compileIntoSPV(
                    compiler,
                    pSourceCode,
                    pSourceCode.size() - 1,
                    shaderKind,
                    pFileName,
                    entryPoint,
                    options
            ));

            long numErrors = shaderc.resultGetNumErrors(result);
            if (numErrors != 0) {
                String errorMessage = Objects.requireNonNull(shaderc.resultGetErrorMessage(result))
                        .readString();
                throw new ShaderCompileException(errorMessage);
            }

            long spvSize = shaderc.resultGetLength(result);
            assert spvSize % 4 == 0 : "SPIR-V size must be a multiple of 4 bytes, got: " + spvSize;

            BytePtr spvBytes = Objects.requireNonNull(shaderc.resultGetBytes(result))
                    .reinterpret(spvSize);
            BytePtr retPtr = BytePtr.allocateAligned(resultArena, spvSize, 4);
            retPtr.segment().copyFrom(spvBytes.segment());
            return retPtr;
        } finally {
            if (result != null) {
                shaderc.resultRelease(result);
            }
        }
    }

    public static ShaderCompiler create(Shaderc shaderc, ShadercUtil.IncludeResolve includeResolve) {
        ShadercCompiler compiler = shaderc.compilerInitialize();
        ShadercCompileOptions options = shaderc.compileOptionsInitialize();

        ShadercUtil.IncludeCallbacks callbacks = ShadercUtil.makeCallbacks(
                Arena.global(),
                includeResolve
        );
        shaderc.compileOptionsSetIncludeCallbacks(
                options,
                callbacks.pfnIncludeResolve,
                callbacks.pfnIncludeResultRelease,
                MemorySegment.NULL
        );

        return new ShaderCompiler(shaderc, compiler, options);
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    private ShaderCompiler(Shaderc shaderc, ShadercCompiler compiler, ShadercCompileOptions options) {
        this.shaderc = shaderc;
        this.compiler = compiler;
        this.options = options;

        this.cleanable = cleaner.register(this, () -> {
            shaderc.compileOptionsRelease(options);
            shaderc.compilerRelease(compiler);
        });
    }

    private final Shaderc shaderc;
    private final ShadercCompiler compiler;
    private final ShadercCompileOptions options;
    private final Cleaner.Cleanable cleanable;

    private static final Cleaner cleaner = Cleaner.create();
    private static final BytePtr entryPoint = BytePtr.allocateString(Arena.global(), "main");
}
