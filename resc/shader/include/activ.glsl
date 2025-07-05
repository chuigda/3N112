#ifndef PR3N112_ACTIV_GLSL
#define PR3N112_ACTIV_GLSL

float sigmoid(float x) {
    // f(x) = 1 / (1 + exp(-x))
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_deriv(float sigmoid_value) {
    // f'(x) = f(x) * (1 - f(x))
    return sigmoid_value * (1.0 - sigmoid_value);
}

float relu(float x) {
    // f(x) = x where x >= 0
    //        0 where x < 0
    return max(0.0, x);
}

float relu_deriv(float relu_value) {
    // f'(x) = 1 where x > 0
    //         0 where x <= 0
    return (relu_value > 0.0) ? 1.0 : 0.0;
}

float leaky_relu(float x) {
    // f(x) = x where x >= 0
    //        0.01 * x where x < 0
    return x < 0.0 ? 0.01 * x : x;
}

float leaky_relu_deriv(float leaky_relu_value) {
    // f'(x) = 1 where x > 0
    //         0.01 where x <= 0
    return (leaky_relu_value > 0.0) ? 1.0 : 0.01;
}

float tanh_deriv(float tanh_value) {
    // f'(x) = 1 - f(x)^2
    return 1.0 - tanh_value * tanh_value;
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
        case ACTIV_TANH:       VALUE = tanh(VALUE);       break; \
        case ACTIV_LINEAR: default: break; \
    }

#define ACTIVATION_DERIVATIVE(MODE_SELECT, VALUE, DERIVATIVE) \
    switch (MODE_SELECT) { \
        case ACTIV_SIGMOID:    DERIVATIVE = sigmoid_deriv(VALUE);    break; \
        case ACTIV_RELU:       DERIVATIVE = relu_deriv(VALUE);       break; \
        case ACTIV_LEAKY_RELU: DERIVATIVE = leaky_relu_deriv(VALUE); break; \
        case ACTIV_TANH:       DERIVATIVE = tanh_deriv(VALUE);       break; \
        case ACTIV_LINEAR: default: DERIVATIVE = 1.0; break; \
    }

#endif // PR3N112_ACTIV_GLSL
