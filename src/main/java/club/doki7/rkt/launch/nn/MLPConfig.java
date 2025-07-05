package club.doki7.rkt.launch.nn;

import java.util.Collections;
import java.util.List;

public final class MLPConfig {
    public static final class LayerConfig {
        public final int size;
        public final Activation activ;

        public LayerConfig(int size, Activation activ) {
            this.size = size;
            this.activ = activ;
        }
    }

    public final int inputSize;
    public final List<LayerConfig> layers;

    public MLPConfig(int inputSize, List<LayerConfig> layers) {
        this.inputSize = inputSize;
        this.layers = Collections.unmodifiableList(layers);
    }
}
