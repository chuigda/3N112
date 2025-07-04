#version 450

layout(constant_id = 0) const uint mnist_image_wh = 28;

const uint input_size = mnist_image_wh * mnist_image_wh;

layout(local_size_x = 1, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer {
    bool mnist_image[input_size];
};
layout(set = 0, binding = 1) buffer WeightsBuffer {
    float weights[];
};
layout(set = 0, binding = 2) buffer BiasBuffer {
    float bias[];
};
layout(set = 0, binding = 3) buffer OutputBuffer {
    float output_data[];
};

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(output_data.length())) {
        return;
    }

    float sum = bias[idx];
    for (int i = 0; i < input_size; ++i) {
        sum += weights[idx * input_size + i] * float(mnist_image[i]);
    }
    output_data[idx] = sigmoid(sum);
}
