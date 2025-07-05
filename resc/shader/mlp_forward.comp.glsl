/// 多层感知机（MLP）前向传播算法
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - activation: 激活函数类型
///   - 0: sigmoid（默认）
///   - 1: identity
///   - 2: relu
///   - 3: leaky relu
///   - 4: tanh
///
/// 配置常量
/// - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个位置开始处理
/// - batch_size: 本批次处理的数据组数
/// - perceptron_count: 感知机的数量，同时也是对批次内每组数据输出的数据数量
/// - input_size: 感知机接受的输入尺寸
/// - use_activation: 本层是否使用激活函数
///
/// 数据
/// - input_data:
///   输入数据，包含所有批次的输入数据
///   本批次（vkCmdDispatch）要处理的数据起始由 input_offset 指定
///   每一批次共处理 batch_size 组输入数据，每组输入数据的大小为 input_size
///   总计为 batch_size * input_size 个 float32
/// - weights: 所有感知机的权重数据，每个感知机的权重数量为 input_size
/// - bias: 所有感知机的偏置数据，每个感知机有一个偏置
/// - output_data: 本批次中所有感知机的输出数据，共计 batch_size * perceptron_count 个 float32

#version 450

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint activation = 0;
layout(local_size_x_id = 0, local_size_y_id = 0) in;

layout(set = 0, binding = 0) uniform Options {
    uint input_offset;
    uint batch_size;
    uint perceptron_count;
    uint input_size;
    bool use_activation;
};

layout(set = 1, binding = 0) buffer InputBuffer {
    float input_data[];
};
layout(set = 1, binding = 1) buffer WeightsBuffer {
    float weights[];
};
layout(set = 1, binding = 2) buffer BiasBuffer {
    float bias[];
};
layout(set = 1, binding = 3) buffer OutputBuffer {
    float output_data[];
};

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float relu(float x) {
    return max(0.0, x);
}

float leaky_relu(float x) {
    return x < 0.0 ? 0.01 * x : x;
}

void main() {
    uint perceptron_index = gl_GlobalInvocationID.x;
    uint batch_index = gl_GlobalInvocationID.y;

    if (perceptron_index >= perceptron_count || batch_index >= batch_size) {
        return;
    }

    uint input_start_index = input_offset + batch_index * input_size;

    float sum = bias[perceptron_index];

    uint weight_start_index = perceptron_index * input_size;
    for (uint i = 0; i < input_size; ++i) {
        sum += input_data[input_start_index + i] * weights[weight_start_index + i];
    }

    if (use_activation) {
        switch (activation) {
            case 0: sum = sigmoid(sum); break;
            case 2: sum = relu(sum); break;
            case 3: sum = leaky_relu(sum); break;
            case 4: sum = tanh(sum); break;
            case 1: default: break;
        }
    }

    uint output_index = batch_index * perceptron_count + perceptron_index;
    output_data[output_index] = sum;
}
