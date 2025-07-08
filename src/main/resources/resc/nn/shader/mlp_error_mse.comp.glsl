/// 计算用于多层感知机（MLP）反向传播的梯度（均方误差算法）
///
/// ## 线程定义
///
/// 每个线程处理本层中 1 个感知机对 1 个样本的梯度计算
/// - gl_GlobalInvocationID.x: 感知机索引
/// - gl_GlobalInvocationID.y: 样本索引
///
/// **注意：这个着色器的线程定义与交叉熵算法（mlp_error_cross_entropy.comp.glsl）不同，需要不同的 dispatch 调用。**
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - perceptron_count: 本层感知机的数量
/// - activation: 本层输出时使用的激活函数类型，参见 include/activ.glsl
///
/// 配置常量
/// - 推理选项（InferOptions）
///   - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个样本开始处理
///   - batch_size: 本批次处理的数据组数
///
/// 输入数据
/// - output_data: 本批次中所有感知机的输出数据，共计 batch_size * perceptron_count 个 float32
/// - expected_output_data: 期望输出数据，包含所有样本的期望输出数据
///   本批次（dispatch）要处理起始样本起始由 input_offset 指定，每个样本对应 perceptron_count 个 float32
///   总计为 batch_size * perceptron_count 个 float32
///
/// 输出数据
/// - gradient_data: 本批次中所有感知机的梯度数据，共计 batch_size * perceptron_count 个 float32

#version 450

#include "include/activ.glsl"

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint perceptron_count = 1;
layout(constant_id = 3) const uint activation = 0;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) uniform InferOptions {
    uint input_offset;
    uint batch_size;
};
layout(set = 0, binding = 1) buffer OutputBuffer {
    readonly float output_data[];
};
layout(set = 0, binding = 2) buffer LabelBuffer {
    readonly float label_data[];
};
layout(set = 0, binding = 3) buffer GradientBuffer {
    writeonly float gradient_data[];
};

void main() {
    const uint perceptron_index = gl_GlobalInvocationID.x;
    const uint sample_index = gl_GlobalInvocationID.y;
    if (sample_index >= batch_size || perceptron_index >= perceptron_count) {
        return;
    }

    const uint label_index = (input_offset + sample_index) * perceptron_count + perceptron_index;
    const uint output_index = sample_index * perceptron_count + perceptron_index;
    const uint gradient_index = output_index;

    const float label = label_data[label_index];
    const float output_value = output_data[output_index];

    const float diff = output_value - label;
    float deriv;
    ACTIVATION_DERIV(activation, output_value, deriv);

    const float gradient_value = diff * deriv;
    gradient_data[gradient_index] = gradient_value;
}
