package club.doki7.rkt.launch.nn;

import java.util.Collections;
import java.util.List;

public final class MLPOptions {
    public static final class Layer {
        public final int size;
        public final Activation activ;
        public final int perceptronWorkgroupSize;

        public Layer(int size, Activation activ, int perceptronWorkgroupSize) {
            this.size = size;
            this.activ = activ;
            this.perceptronWorkgroupSize = perceptronWorkgroupSize;
        }
    }

    public final int inputSize;
    public final List<Layer> layers;
    public final boolean useSharedMemory;

    public MLPOptions(
            int inputSize,
            List<Layer> layers,
            boolean useSharedMemory
    ) {
        this.inputSize = inputSize;
        this.layers = Collections.unmodifiableList(layers);
        this.useSharedMemory = useSharedMemory;
    }
}
