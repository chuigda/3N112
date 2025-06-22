package club.doki7.rkt.img;

import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.ffm.ptr.IntPtr;
import club.doki7.stb.STBJavaTraceUtil;
import club.doki7.stb.image.STBI;
import club.doki7.stb.image.STBIUtil;
import club.doki7.stb.image.datatype.STBI_IoCallbacks;
import org.jetbrains.annotations.NonNls;
import org.jetbrains.annotations.NotNull;

import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/// Image reading.
/// For full image manipulation, use {@link ImageManipulator} instead.
public class ImageLoader {
    public STBI stbI;

    public ImageLoader(ISharedLibrary libSTB) {
        STBJavaTraceUtil.enableJavaTraceForSTB(libSTB);
        stbI = new STBI(libSTB);
    }

    /// @param desiredChannels 0 for auto-detect, otherwise # of image components requested in result
    public @NotNull CPUImage fromFile(@NotNull RandomAccessFile file, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            STBI_IoCallbacks fileCallback = STBIUtil.makeIOCallbacks(arena, file);
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr image = stbI.loadFromCallbacks(fileCallback, MemorySegment.NULL, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read());
        }
    }

    public @NotNull CPUImage fromPath(@NotNull @NonNls String path, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            BytePtr pPath = BytePtr.allocateString(arena, path);
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr image = stbI.load(pPath, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read());
        }
    }

    public @NotNull CPUImage fromBytes(byte @NotNull [] data, int desiredChannels) {
        try (Arena arena = Arena.ofConfined()) {
            IntPtr pWidth = IntPtr.allocate(arena);
            IntPtr pHeight = IntPtr.allocate(arena);
            IntPtr pNumChannels = IntPtr.allocate(arena);
            BytePtr pData = BytePtr.allocate(arena, data);
            BytePtr image = stbI.loadFromMemory(pData, data.length, pWidth, pHeight, pNumChannels, desiredChannels);
            return new CPUImage(image, pWidth.read(), pHeight.read(), pNumChannels.read());
        }
    }
}
