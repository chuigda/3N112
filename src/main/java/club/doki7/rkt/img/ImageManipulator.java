package club.doki7.rkt.img;

import club.doki7.ffm.library.ISharedLibrary;
import club.doki7.ffm.ptr.BytePtr;
import club.doki7.rkt.util.ExcUtil;
import club.doki7.stb.imageresize.STBIR;
import club.doki7.stb.imageresize.enumtype.STBIR_PixelLayout;
import club.doki7.stb.imagewrite.STBIW;
import club.doki7.stb.imagewrite.STBWUtil;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/// Full image manipulation, requires additional native methods being registered.
public class ImageManipulator extends ImageLoader {
    public STBIR stbIR;
    public STBIW stbIW;

    public ImageManipulator(ISharedLibrary libSTB) {
        super(libSTB);
        stbIR = new STBIR(libSTB);
        stbIW = new STBIW(libSTB);
    }

    public void writePngPath(@NotNull Image image, @NotNull String path) {
        try (Arena arena = Arena.ofConfined()) {
            stbIW.writePng(BytePtr.allocateString(arena, path),
                image.width(), image.height(), image.numChannels(),
                image.data().segment(),
                image.width() * image.numChannels());
        }
    }

    public void writePngStream(@NotNull Image image, @NotNull OutputStream stream) {
        try (Arena arena = Arena.ofConfined()) {
            stbIW.writePngToFunc(
                STBWUtil.makeWriteCallback(arena, ExcUtil.sneakyConsumer(segment ->
                    stream.write(segment.asByteBuffer().array()))),
                MemorySegment.NULL,
                image.width(), image.height(), image.numChannels(),
                image.data().segment(),
                image.width() * image.numChannels());
        }
    }

    public @NotNull Image resize(@NotNull Image image, int newWidth, int newHeight) {
        if (newWidth == image.width() && newHeight == image.height()) {
            return image; // No resize needed
        }
        try (Arena arena = Arena.ofConfined()) {
            BytePtr resizedData = BytePtr.allocate(arena, (long) newWidth * newHeight * image.numChannels());
            stbIR.resizeUint8Srgb(image.data(), image.width(), image.height(),
                image.numChannels(), resizedData, newWidth, newHeight,
                newWidth * image.numChannels(), STBIR_PixelLayout.RGBA);
            return new Image(resizedData, newWidth, newHeight, image.numChannels());
        }
    }
}
