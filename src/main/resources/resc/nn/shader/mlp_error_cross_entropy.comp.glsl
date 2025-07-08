/// 计算用于多层感知机（MLP）反向传播的梯度（交叉熵误差算法）
///
/// ## 线程定义
///
/// 每个线程处理本层中所有感知机对 1 个样本的误差计算
/// - gl_GlobalInvocationID.x: 样本索引
///
/// **注意：这个着色器的线程定义与 MSE 算法（mlp_error_mse.comp.glsl）不同，需要不同的 dispatch 调用。**
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - perceptron_count: 本层感知机的数量，也就是输出数据的大小
///
/// **注意：这个着色器不能用于输出规模较大的感知机层，会导致性能退化甚至编译失败。**
/// **好在大多数情况下，使用 Softmax/交叉熵算法的感知机层规模都不会太大。**
///
/// 配置常量
/// - 推理选项（InferOptions）
///   - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个样本开始处理
///     在误差计算中，这个偏移量同时也指定从期望输出数据（expected_output_data）开始的偏移量
///   - batch_size: 本批次处理的数据组数
///
/// 输入数据
/// - output_data: 本批次中所有感知机的输出数据，共计 batch_size * perceptron_count 个 float32
/// - expected_output_data: 期望的输出数据，共计 batch_size 个 uint
///
/// 输出数据
/// - gradient_data: 本批次中所有感知机的梯度数据，共计 batch_size * perceptron_count 个 float32

#version 450

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint perceptron_count = 1;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) uniform InferOptions {
    uint input_offset;
    uint batch_size;
};
layout(set = 0, binding = 1) buffer OutputBuffer {
    readonly float output_data[];
};
layout(set = 0, binding = 2) buffer ExpectedOutputBuffer {
    readonly uint expected_output_data[];
};
layout(set = 0, binding = 3) buffer GradientBuffer {
    writeonly float gradient_data[];
};

void main() {
    const uint sample_index = gl_GlobalInvocationID.x;
    if (sample_index >= batch_size) {
        return;
    }

    const uint expected_output_start_index = input_offset + sample_index;
    const uint expected_value = expected_output_data[expected_output_start_index];

    const uint output_start_index = sample_index * perceptron_count;
    const uint gradient_start_index = output_start_index;

    float cached_output_data[perceptron_count];
    float max_output_value = output_data[output_start_index];
    for (uint i = 0; i < perceptron_count; ++i) {
        float output_value = output_data[output_start_index + i];
        cached_output_data[i] = output_value;
        if (output_value > max_output_value) {
            max_output_value = output_value;
        }
    }

    float sum_exp = 0.0;
    float output_exp[perceptron_count];
    for (uint i = 0; i < perceptron_count; ++i) {
        cached_output_data[i] -= max_output_value;
        const float output_value = cached_output_data[i];
        const float exp_value = exp(output_value);
        output_exp[i] = exp_value;
        sum_exp += exp_value;
    }

    for (uint i = 0; i < perceptron_count; ++i) {
        const float exp_value = output_exp[i];
        const float softmax_value = exp_value / sum_exp;

        const float gradient = softmax_value - (i == expected_value ? 1.0 : 0.0);
        gradient_data[gradient_start_index + i] = gradient;
    }
}
