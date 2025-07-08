/// 多层感知机（MLP）权重预热算法
///
/// ## 线程定义
///
/// 每个线程处理 1 个感知机的 1 个权重的预热
/// - gl_GlobalInvocationID.x: 感知机索引
/// - gl_GlobalInvocationID.y: 权重索引
/// - gl_NumWorkGroups.x * gl_WorkGroupSize.x: 感知机总数
/// - gl_NumWorkGroups.y * gl_WorkGroupSize.y: 每个感知机的权重数量
///
/// ## 参数定义
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - perceptron_count: 本层感知机的数量
/// - input_size: 每个感知机接受的输入数据大小
/// - activation: 激活函数类型，参见 include/activ.glsl
///   这个参数会决定预热的方式（Sigmoid, Tanh 和 Linear 用 Xavier，RELU 和 Leaky RELU 用 He）
///
/// 推送常量
/// - seed: 随机种子，用于生成权重和偏置的初始值
///
/// 输出数据
/// - weights: 所有感知机的权重数据，每个感知机的权重数量为 input_size
///   共计 perceptron_count * input_size 个 float32
/// - bias: 所有感知机的偏置数据，每个感知机有一个偏置
///   共计 perceptron_count 个 float32

#version 450

#include "include/activ.glsl"

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint perceptron_count = 1;
layout(constant_id = 3) const uint input_size = 1;
layout(constant_id = 4) const uint activation = 0;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) buffer WeightsBuffer {
    writeonly float weights[];
};
layout(set = 0, binding = 1) buffer BiasesBuffer {
    writeonly float biases[];
};
layout(push_constant) uniform PushConstants {
    float seed;
};

float random(vec2 xy) {
    return 2.0 * fract(sin(dot(xy + seed, vec2(12.9898, 78.233))) * 43758.5453123) - 1.0;
}

void main() {
    const uvec2 id = gl_GlobalInvocationID.xy;
    const uint perceptron_id = id.x;
    const uint weight_id = id.y;

    const uint weight_index = perceptron_id * input_size + weight_id;
    const uint bias_index = perceptron_id;

    const float rand_val = random(vec2(id));
    float std_dev;
    switch (activation) {
        case ACTIV_RELU:
        case ACTIV_LEAKY_RELU: {
            std_dev = sqrt(2.0 / float(input_size));
            break;
        }
        case ACTIV_SIGMOID:
        case ACTIV_TANH:
        case ACTIV_LINEAR:
        default: {
            std_dev = sqrt(1.0 / float(input_size));
            break;
        }
    }
    weights[weight_index] = rand_val * std_dev;

    if (weight_id == 0) {
        biases[bias_index] = 0.0;
    }
}
