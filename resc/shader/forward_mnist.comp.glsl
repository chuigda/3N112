#version 450

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(local_size_x_id = 0, local_size_y_id = 0) in;

layout(set = 0, binding = 0) buffer InputBuffer {
    float input_data[];
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

layout(push_constant) uniform PushConstants {
    uint input_size;
    bool use_activation;
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
        sum += weights[idx * input_size + i] * input_data[i];
    }

    if (use_activation) {
        output_data[idx] = sigmoid(sum);
    } else {
        output_data[idx] = sum;
    }
}
