/// 多层感知机（MLP）前向传播算法
///
/// 特化常量：
/// - tx, ty: 优化选项，指定工作组的大小
///
/// 配置常量
/// - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个位置开始处理
/// - input_size: 感知机接受的输入尺寸
/// - use_activation: 本层是否使用激活函数
///
/// 数据
/// - input_data: 输入数据，包含所有批次的输入数据
///               本批次（vkCmdDispatch）要处理的数据起始由 input_offset 指定
///               每一批次共处理 input_size * batch_size 个数据
/// - weights: 所有感知机的权重数据，每个感知机的权重数量为 input_size
/// - bias: 所有感知机的偏置数据，每个感知机有一个偏置
/// - output_data: 本批次中所有感知机的输出数据，总量为 batch_size * perceptron_count

#version 450

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(local_size_x_id = 0, local_size_y_id = 0) in;

layout(set = 0, binding = 0) uniform Options {
    uint input_offset;
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

void main() {
    uint perceptron_count = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    uint batch_size = gl_WorkGroupSize.y * gl_NumWorkGroups.y;

    uint perceptron_index = gl_GlobalInvocationID.x;
    uint batch_index = gl_GlobalInvocationID.y;
}

