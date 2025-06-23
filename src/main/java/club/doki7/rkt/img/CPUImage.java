package club.doki7.rkt.img;

import club.doki7.ffm.ptr.BytePtr;

import java.lang.ref.Cleaner;

/// @see CPUImageUtil
public final class CPUImage implements AutoCloseable {
    public final BytePtr data;
    public final int width;
    public final int height;
    public final int numChannels;

    CPUImage(BytePtr data, int width, int height, int numChannels, CPUImageUtil u) {
        this.data = data.reinterpret((long) width * height * numChannels);
        this.width = width;
        this.height = height;
        this.numChannels = numChannels;

        this.cleanable = CPUImageUtil.cleaner.register(
                this,
                () -> u.stbI.imageFree(data.segment())
        );
    }

    @Override
    public void close() throws Exception {
        this.cleanable.clean();
    }

    private final Cleaner.Cleanable cleanable;
}
