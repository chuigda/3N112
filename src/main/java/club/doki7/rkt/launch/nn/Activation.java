package club.doki7.rkt.launch.nn;

public enum Activation {
    SIGMOID(0),
    LINEAR(1),
    RELU(2),
    LEAKY_RELU(3),
    TANH(4);

    public final int value;

    Activation(int value) {
        this.value = value;
    }
}
