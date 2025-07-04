package club.doki7.rkt.launch.nn;

import java.util.List;

public final class MLPConfig {
    public final int inputSize;
    public final List<Integer> hiddenLayersSize;

    public MLPConfig(int inputSize, List<Integer> hiddenLayersSize) {
        this.inputSize = inputSize;
        this.hiddenLayersSize = hiddenLayersSize;
    }
}
