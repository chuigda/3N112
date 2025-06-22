package club.doki7.rkt.img;

import club.doki7.ffm.ptr.BytePtr;
import org.jetbrains.annotations.NotNull;

/// @see ImageLoader
/// @see ImageManipulator
public record CPUImage(
    @NotNull BytePtr data,
    int width,
    int height,
    int numChannels
) {
}
