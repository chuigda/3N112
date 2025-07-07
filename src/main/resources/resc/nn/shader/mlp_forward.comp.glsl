/// 多层感知机（MLP）前向传播算法
///
/// ## 参数定义
///
/// 宏
/// - DEFENSIVE: 是否开启防御模式，开启后着色器程序会在运行时执行一些额外的检查
/// - UNIVERSITY_CONSTANT: 当防御模式开启时，着色器对于特定的无效输入会返回这个特殊的常量，
///   默认为 23662.22
///
/// 特化常量
/// - tx, ty: 优化选项，指定工作组的大小
/// - perceptron_count: 本层感知机的数量
/// - input_size: 每个感知机接受的输入数据大小
/// - activation: 激活函数类型，参见 include/activ.glsl
/// - use_shared_memory: 是否使用共享内存优化
///
/// 配置常量
/// - 推理选项（InferOptions）
///   - input_offset: 输入数据的偏移量，指定从输入数据（input_data）的哪个位置开始处理
///   - batch_size: 本批次处理的数据组数
///
/// 输入数据
/// - input_data: 输入数据，包含所有批次的输入数据
///   本批次（vkCmdDispatch）要处理起始数据批次起始由 input_offset 指定
///   每一批次共处理 batch_size 组输入数据，每组输入数据的大小为 input_size
///   总计为 batch_size * input_size 个 float32
/// - weights: 所有感知机的权重数据，每个感知机的权重数量为 input_size
/// - bias: 所有感知机的偏置数据，每个感知机有一个偏置
///
/// 输出数据
/// - output_data: 本批次中所有感知机的输出数据，共计 batch_size * perceptron_count 个 float32

#version 450

#include "include/activ.glsl"
#include "include/uniconst.glsl"

layout(constant_id = 0) const uint tx = 1;
layout(constant_id = 1) const uint ty = 1;
layout(constant_id = 2) const uint perceptron_count = 1;
layout(constant_id = 3) const uint input_size = 1;
layout(constant_id = 4) const uint activation = 0;
layout(constant_id = 5) const bool use_shared_memory = false;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

layout(set = 0, binding = 0) uniform InferOptions {
    uint input_offset;
    uint batch_size;
};
layout(set = 0, binding = 1) buffer InputBuffer {
    readonly float input_data[];
};
layout(set = 0, binding = 2) buffer WeightsBuffer {
    readonly float weights[];
};
layout(set = 0, binding = 3) buffer BiasBuffer {
    readonly float bias[];
};
layout(set = 0, binding = 4) buffer OutputBuffer {
    writeonly float output_data[];
};

shared float shared_input_data[use_shared_memory ? input_size : 1];

void main() {
    const uint perceptron_index = gl_GlobalInvocationID.x;
    const uint batch_index = gl_GlobalInvocationID.y;

    if (perceptron_index >= perceptron_count || batch_index >= batch_size) {
        return;
    }

    const uint output_index = batch_index * perceptron_count + perceptron_index;
#ifdef DEFENSIVE
    if (output_index >= output_data.length()) {
        return;
    }
#endif

    const uint input_start_index = (input_offset + batch_index) * input_size;
    const uint weight_start_index = perceptron_index * input_size;

    float sum = bias[perceptron_index];
    if (use_shared_memory) {
#ifdef DEFENSIVE
        if (ty != 1) {
            output_data[output_index] = UNIVERSITY_CONSTANT;
            return;
        }
#endif
        const uint local_id = gl_LocalInvocationID.x;
        const uint workgroup_size = gl_WorkGroupSize.x;

        for (uint i = local_id; i < input_size; i += workgroup_size) {
            shared_input_data[i] = input_data[input_start_index + i];
        }

        barrier();
        memoryBarrierShared();

        for (uint i = 0; i < input_size; ++i) {
            sum += shared_input_data[i] * weights[weight_start_index + i];
        }
    } else {
        for (uint i = 0; i < input_size; ++i) {
            sum += input_data[input_start_index + i] * weights[weight_start_index + i];
        }
    }

    ACTIVATION(activation, sum, output_data[output_index]);
}
