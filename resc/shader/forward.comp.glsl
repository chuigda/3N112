#version 450

layout(local_size_x = 1, local_size_y = 1) in;

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
    output_data[idx] = sigmoid(sum);
}
