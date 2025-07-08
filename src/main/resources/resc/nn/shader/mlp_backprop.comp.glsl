/// 为一个隐藏层计算反向传播的梯度
///
/// ## 线程定义
///
/// 每个线程处理本层中 1 个感知机对 1 个样本的梯度计算。
/// - gl_GlobalInvocationID.x: 本层感知机的索引 (j)
/// - gl_GlobalInvocationID.y: 样本索引
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - perceptron_count: 本层感知机的数量
/// - next_perceptron_count: 下一层感知机的数量
/// - activation: 本层输出时使用的激活函数类型，参见 include/activ.glsl
///
/// 配置常量
/// - 推理选项（InferOptions）
///   - input_offset: 未使用，但为了描述符集兼容性而保留
///   - batch_size: 本批次处理的数据组数
///
/// 输入数据
/// - next_layer_gradient_data: 下一层的梯度数据，共计 batch_size * next_perceptron_count 个 float32
/// - next_layer_weights_data: 下一层的权重数据，一个 next_perceptron_count * perceptron_count 的矩阵
///   （连接本层和下一层）
/// - current_layer_output_data: 本层的输出数据，用于计算激活函数的导数，共计 batch_size * perceptron_count 个 float32
///
/// 输出数据
/// - gradient_data: 计算出的本层梯度数据，共计 batch_size * perceptron_count 个 float32

#version 450

#include "include/activ.glsl"

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint perceptron_count = 1;
layout(constant_id = 3) const uint next_perceptron_count = 1;
layout(constant_id = 4) const uint activation = 0;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) uniform InferOptions {
    uint input_offset;
    uint batch_size;
};
layout(set = 0, binding = 1) buffer NextLayerGradientBuffer {
    readonly float next_layer_gradient_data[];
};
layout(set = 0, binding = 2) buffer NextLayerWeightsBuffer {
    readonly float next_layer_weights_data[];
};
layout(set = 0, binding = 3) buffer OutputBuffer {
    readonly float output_data[];
};
layout(set = 0, binding = 4) buffer GradientBuffer {
    writeonly float gradient_data[];
};

void main() {
    const uint current_perceptron_index = gl_GlobalInvocationID.x;
    const uint sample_index = gl_GlobalInvocationID.y;

    if (sample_index >= batch_size || current_perceptron_index >= perceptron_count) {
        return;
    }

    float weighted_error_sum = 0.0;
    for (uint k = 0; k < next_perceptron_count; ++k) {
        const float next_layer_error = next_layer_gradient_data[sample_index * next_perceptron_count + k];

        const uint weight_index = k * perceptron_count + current_perceptron_index;
        const float weight_jk = next_layer_weights_data[weight_index];

        weighted_error_sum += next_layer_error * weight_jk;
    }

    const uint output_index = sample_index * perceptron_count + current_perceptron_index;
    const float output_value = output_data[output_index];
    float deriv;
    ACTIVATION_DERIV(activation, output_value, deriv);

    const float new_gradient = weighted_error_sum * deriv;
    const uint gradient_index = output_index;
    gradient_data[gradient_index] = new_gradient;
}
