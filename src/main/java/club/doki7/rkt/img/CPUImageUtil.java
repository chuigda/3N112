package club.doki7.rkt.img;

import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.rkt.util.ExcUtil;
import club.doki7.stb.STBJavaTraceUtil;
import club.doki7.stb.image.STBI;
import club.doki7.stb.image.STBIUtil;
import club.doki7.stb.image.datatype.STBI_IoCallbacks;
import club.doki7.stb.imageresize.STBIR;
import club.doki7.stb.imageresize.enumtype.STBIR_PixelLayout;
import club.doki7.stb.imagewrite.STBIW;
import club.doki7.stb.imagewrite.STBWUtil;
import org.jetbrains.annotations.NonNls;
import org.jetbrains.annotations.NotNull;

import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;

public final class CPUImageUtil {
    public final STBI stbI;
    public final STBIR stbIR;
    public final STBIW stbIW;

    public CPUImageUtil(ISharedLibrary libSTB) {
        STBJavaTraceUtil.enableJavaTraceForSTB(libSTB);
        stbI = new STBI(libSTB);
        stbIR = new STBIR(libSTB);
        stbIW = new STBIW(libSTB);
    }

    /// @param desiredChannels 0 for auto-detect, otherwise # of image components requested in result
    public @NotNull CPUImage fromFile(@NotNull RandomAccessFile file, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            STBI_IoCallbacks fileCallback = STBIUtil.makeIOCallbacks(arena, file);
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr image = stbI.loadFromCallbacks(fileCallback, MemorySegment.NULL, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read(), this);
        }
    }

    public @NotNull CPUImage fromPath(@NotNull @NonNls String path, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            BytePtr pPath = BytePtr.allocateString(arena, path);
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr image = stbI.load(pPath, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read(), this);
        }
    }

    public @NotNull CPUImage fromBytes(byte @NotNull [] data, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr pData = BytePtr.allocate(arena, data);
            BytePtr image = stbI.loadFromMemory(pData, data.length, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read(), this);
        }
    }

    public void writePngPath(@NotNull CPUImage image, @NotNull String path) {
        try (Arena arena = Arena.ofConfined()) {
            stbIW.writePng(
                    BytePtr.allocateString(arena, path),
                    image.width, image.height, image.numChannels,
                    image.data.segment(),
                    image.width * image.numChannels
            );
        }
    }

    public void writePngStream(@NotNull CPUImage image, @NotNull OutputStream stream) {
        try (Arena arena = Arena.ofConfined()) {
            stbIW.writePngToFunc(
                    STBWUtil.makeWriteCallback(arena, ExcUtil.sneakyConsumer(segment ->
                            stream.write(segment.asByteBuffer().array()))),
                    MemorySegment.NULL,
                    image.width, image.height, image.numChannels,
                    image.data.segment(),
                    image.width * image.numChannels
            );
        }
    }

    public @NotNull CPUImage resize(@NotNull CPUImage image, int newWidth, int newHeight) {
        try (Arena arena = Arena.ofConfined()) {
            BytePtr resizedData = BytePtr.allocate(arena, (long) newWidth * newHeight * image.numChannels);
            stbIR.resizeUint8Srgb(
                    image.data,
                    image.width, image.height, image.width * image.numChannels,
                    resizedData,
                    newWidth, newHeight, newWidth * image.numChannels,
                    STBIR_PixelLayout.RGBA
            );
            return new CPUImage(resizedData, newWidth, newHeight, image.numChannels, this);
        }
    }

    static final Cleaner cleaner = Cleaner.create();
}
