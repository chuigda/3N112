#ifndef PR3N112_ACTIV_GLSL
#define PR3N112_ACTIV_GLSL

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float relu(float x) {
    return max(0.0, x);
}

float leaky_relu(float x) {
    return x < 0.0 ? 0.01 * x : x;
}

#define ACTIV_SIGMOID    0
#define ACTIV_DEFAULT    ACTIV_SIGMOID
#define ACTIV_LINEAR     1
#define ACTIV_RELU       2
#define ACTIV_LEAKY_RELU 3
#define ACTIV_TANH       4

#define ACTIVATION(MODE_SELECT, VALUE) \
    switch (MODE_SELECT) { \
        case ACTIV_SIGMOID:    VALUE = sigmoid(VALUE);    break; \
        case ACTIV_RELU:       VALUE = relu(VALUE);       break; \
        case ACTIV_LEAKY_RELU: VALUE = leaky_relu(VALUE); break; \
        case ACTIV_TANH:       VALUE = tanh(VALUE);      break; \
        case ACTIV_LINEAR: default: break; \
    }

#endif // PR3N112_ACTIV_GLSL
