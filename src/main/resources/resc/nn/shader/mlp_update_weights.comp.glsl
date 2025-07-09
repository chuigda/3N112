/// 更新多层感知机（MLP）一层的权重和偏置
///
/// ## 线程定义
///
/// 每个线程更新 1 个权重。
/// - gl_GlobalInvocationID.x: 前一层神经元的索引（即输入特征的索引）
/// - gl_GlobalInvocationID.y: 本层神经元的索引
///
/// 此外，每个神经元对应的第一个线程（input_index == 0）还将负责更新该神经元的偏置。
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - input_size: 本层的输入规模（即前一层的神经元数量）
/// - perceptron_count: 本层的感知机数量
///
/// 配置常量
/// - 更新选项（UpdateOptions）
///   - learning_rate: 学习率
///   - batch_size: 本批次处理的数据组数
/// - 推理选项（InferOptions）
///   - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个样本开始处理
///   - batch_size_dup: 未使用，但为了描述符集兼容性而保留
///
/// 输入数据
/// - input_data: 本层的输入数据（即前一层的输出数据），共计 batch_size * input_size 个 float32
/// - gradient_data: 本层的梯度数据（由 mlp_error_*.comp.glsl 或 mlp_backprop_hidden.comp.glsl 计算得出）
///   共计 batch_size * perceptron_count 个 float32
///
/// 输入/输出数据 (读写)
/// - weights_data: 本层的权重数据，一个 input_size * perceptron_count 的矩阵
/// - biases_data: 本层的偏置数据，一个包含 perceptron_count 个元素的向量

#version 450

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint input_size = 1;
layout(constant_id = 3) const uint perceptron_count = 1;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) uniform UpdateOptions {
    float learning_rate;
    uint batch_size;
};
layout(set = 0, binding = 1) uniform InferOptions {
    uint input_offset;
    uint batch_size_dup; // unused, for descriptor set compatibility
};
layout(set = 0, binding = 2) buffer InputBuffer {
    readonly float input_data[];
};
layout(set = 0, binding = 3) buffer GradientBuffer {
    readonly float gradient_data[];
};
layout(set = 0, binding = 4) buffer WeightsBuffer {
    float weights[];
};
layout(set = 0, binding = 5) buffer BiasesBuffer {
    float biases[];
};

void main() {
    const uint input_index = gl_GlobalInvocationID.x;
    const uint perceptron_index = gl_GlobalInvocationID.y;

    if (input_index >= input_size || perceptron_index >= perceptron_count) {
        return;
    }

    const uint input_start_index = input_offset * input_size + input_index;
    float weight_gradient_sum = 0.0;
    for (uint sample_index = 0; sample_index < batch_size; ++sample_index) {
        const float input_value = input_data[input_start_index + sample_index * input_size];
        const float error_signal = gradient_data[sample_index * perceptron_count + perceptron_index];
        weight_gradient_sum += input_value * error_signal;
    }

    const float avg_weight_gradient = weight_gradient_sum / float(batch_size);

    const uint weight_index = perceptron_index * input_size + input_index;
    weights[weight_index] -= learning_rate * avg_weight_gradient;

    if (input_index == 0) {
        float bias_gradient_sum = 0.0;
        for (uint sample_index = 0; sample_index < batch_size; ++sample_index) {
            const float error_signal = gradient_data[sample_index * perceptron_count + perceptron_index];
            bias_gradient_sum += error_signal;
        }

        const float avg_bias_gradient = bias_gradient_sum / float(batch_size);
        biases[perceptron_index] -= learning_rate * avg_bias_gradient;
    }
}
