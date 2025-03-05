#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 添加half精度支持
#include "radixSelectTopK_thrust.cuh"

// 比较函数用于验证结果
bool greater_than(half a, half b)
{
    return __half2float(a) > __half2float(b); // 转换为float进行比较
}

// 用于排序的pair结构
struct ValueIndexPair {
    half value;
    int index;
    
    bool operator<(const ValueIndexPair& other) const {
        return __half2float(value) > __half2float(other.value); // 降序排列，需要转换为float比较
    }
};

// 验证结果是否正确
bool validateResults(half* h_output, uint* h_indices, ValueIndexPair* h_reference_pairs, half* h_input, int k) {
    bool success = true;
    
    // 创建临时数组来储存GPU结果的值-索引对，用于排序
    ValueIndexPair *gpu_pairs = new ValueIndexPair[k];
    for (int i = 0; i < k; i++) {
        gpu_pairs[i].value = h_output[i];
        gpu_pairs[i].index = h_indices[i];
    }
    
    // 排序GPU结果
    std::sort(gpu_pairs, gpu_pairs + k);
    
    // 验证值和索引
    for (int i = 0; i < k; i++) {
        // 验证值是否正确 (使用更大的容差，因为FP16精度较低)
        if (std::abs(__half2float(gpu_pairs[i].value) - __half2float(h_reference_pairs[i].value)) > 1e-2) {
            printf("错误: 位置 %d, GPU值: %f, CPU值: %f\n", 
                   i, __half2float(gpu_pairs[i].value), __half2float(h_reference_pairs[i].value));
            success = false;
            break;
        }
        
        // 验证索引是否正确（通过检查该索引在原始数组中的值是否等于当前值）
        if (std::abs(__half2float(h_input[gpu_pairs[i].index]) - __half2float(gpu_pairs[i].value)) > 1e-2) {
            printf("索引错误: 位置 %d, 索引 %d, 索引对应的原始值: %f, 期望值: %f\n", 
                   i, gpu_pairs[i].index, __half2float(h_input[gpu_pairs[i].index]), 
                   __half2float(gpu_pairs[i].value));
            success = false;
            break;
        }
    }
    
    delete[] gpu_pairs;
    return success;
}

// 测试radixSelectTopKThrustK方法
float testRadixSelectTopKThrust(half* h_input, half* h_output, uint* h_indices, 
                        half* d_input, half* d_output, uint* d_indices, 
                        ValueIndexPair* h_reference_pairs, 
                        int num_items, int k, int run, bool validate) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 将数据复制到设备
    cudaMemcpy(d_input, h_input, num_items * sizeof(half), cudaMemcpyHostToDevice);

    // 计时开始
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    // 执行TopK
    cudaError_t error = radixSelectTopKThrust<half>(d_input, num_items, k, d_output, d_indices);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 等待GPU完成
    cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1.0f;
    }
    
    float kernel_milliseconds = 0;
    cudaEventElapsedTime(&kernel_milliseconds, start, stop);
    
    cudaMemcpy(h_output, d_output, k * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, k * sizeof(uint), cudaMemcpyDeviceToHost);

    if (run % 10 == 0) {
        printf("radixSelectTopK GPU计算的前10个值: ");
        for (int i = 0; i < std::min(10, k); i++) {
            printf("%f(索引%d) ", __half2float(h_output[i]), h_indices[i]);
        }
        printf("\n");
    }

    // 验证结果
    if (validate) {
        bool success = validateResults(h_output, h_indices, h_reference_pairs, h_input, k);
        if (success) {
            printf("验证成功! radixSelectTopK 结果与索引均正确\n");
        } else {
            printf("验证失败! radixSelectTopK 结果或索引不正确\n");
        }
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return kernel_milliseconds;
}

// 测试radixSelectTopKPreallocated方法
float testRadixSelectTopKPreallocatedThrust(half* h_input, half* h_output, uint* h_indices, 
                                    half* d_input, half* d_output, uint* d_indices, 
                                    ValueIndexPair* h_reference_pairs, 
                                    int num_items, int k, int run, bool validate) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预分配内存
    half* d_keys_sorted = nullptr;
    uint* d_indices_in = nullptr;
    cudaMalloc(&d_keys_sorted, num_items * sizeof(half));
    cudaMalloc(&d_indices_in, num_items * sizeof(uint));
    
    // 将数据复制到设备
    cudaMemcpy(d_input, h_input, num_items * sizeof(half), cudaMemcpyHostToDevice);

    // 计时开始
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    // 执行TopK
    cudaError_t error = radixSelectTopKPreallocatedThrust<half>(
        d_input, num_items, k, d_output, d_indices, d_keys_sorted, d_indices_in);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 等待GPU完成
    cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        cudaFree(d_keys_sorted);
        cudaFree(d_indices_in);
        return -1.0f;
    }
    
    float kernel_milliseconds = 0;
    cudaEventElapsedTime(&kernel_milliseconds, start, stop);
    
    cudaMemcpy(h_output, d_output, k * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, k * sizeof(uint), cudaMemcpyDeviceToHost);

    if (run % 10 == 0) {
        printf("radixSelectTopKPreallocated GPU计算的前10个值: ");
        for (int i = 0; i < std::min(10, k); i++) {
            printf("%f(索引%d) ", __half2float(h_output[i]), h_indices[i]);
        }
        printf("\n");
    }

    // 验证结果
    if (validate) {
        bool success = validateResults(h_output, h_indices, h_reference_pairs, h_input, k);
        if (success) {
            printf("验证成功! radixSelectTopKPreallocated 结果与索引均正确\n");
        } else {
            printf("验证失败! radixSelectTopKPreallocated 结果或索引不正确\n");
        }
    }
    
    // 释放预分配的内存
    cudaFree(d_keys_sorted);
    cudaFree(d_indices_in);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return kernel_milliseconds;
}

int main()
{
    // 测试参数
    const int num_items = 21600;
    const int k = 1024;
    const int num_runs = 100;
    
    // 选择是否验证结果（每10次运行验证一次）
    const bool validate = true;

    // 分配主机内存
    half *h_input = new half[num_items];
    half *h_output = new half[k];
    uint *h_indices = new uint[k];
    ValueIndexPair *h_reference_pairs = new ValueIndexPair[num_items]; // 用于验证的参考结果

    // 分配设备内存
    half *d_input, *d_output;
    uint *d_indices;
    cudaMalloc(&d_input, num_items * sizeof(half));
    cudaMalloc(&d_output, k * sizeof(half));
    cudaMalloc(&d_indices, k * sizeof(uint));

    // 性能记录
    float total_time_standard = 0.0f;
    float total_time_preallocated = 0.0f;
    
    printf("开始测试 radixSelectTopK 和 radixSelectTopKPreallocated...\n");
    printf("参数: 元素总数 = %d, k = %d, 运行次数 = %d\n", num_items, k, num_runs);
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int run = 0; run < num_runs; run++)
    {
        if (run % 10 == 0) {
            printf("\n运行 %d/%d\n", run+1, num_runs);
            printf("生成随机数据...\n");
        }
        
        // 生成随机数据
        for (int i = 0; i < num_items; i++)
        {
            float rand_val = dist(gen);
            h_input[i] = __float2half(rand_val); // 转换float到half
            
            // 确保有一些已知的大值进行测试
            if (i < 10) {
                h_input[i] = __float2half(0.9f + i * 0.01f);  // 前10个元素为0.90, 0.91, 0.92...0.99
            }
            
            h_reference_pairs[i].value = h_input[i];
            h_reference_pairs[i].index = i;
        }

        // 只在每10次运行时打印数据集统计信息
        if (run % 10 == 0) {
            half min_val = h_input[0], max_val = h_input[0];
            float sum_val = 0;
            for (int i = 0; i < num_items; i++) {
                if (__half2float(h_input[i]) < __half2float(min_val))
                    min_val = h_input[i];
                if (__half2float(h_input[i]) > __half2float(max_val))
                    max_val = h_input[i];
                sum_val += __half2float(h_input[i]);
            }
            printf("数据集统计: 最小值=%f, 最大值=%f, 平均值=%f\n", 
                   __half2float(min_val), __half2float(max_val), sum_val/num_items);
        }

        // 在CPU上计算参考结果
        std::sort(h_reference_pairs, h_reference_pairs + num_items);

        if (run % 10 == 0) {
            printf("CPU排序后的前10个值: ");
            for (int i = 0; i < std::min(10, k); i++) {
                printf("%f ", __half2float(h_reference_pairs[i].value));
            }
            printf("\n");
        }

        // 测试标准版本
        float time_standard = testRadixSelectTopKThrust(h_input, h_output, h_indices, 
                                               d_input, d_output, d_indices, 
                                               h_reference_pairs, num_items, k, run, 
                                               validate && (run % 10 == 0));
        total_time_standard += time_standard;
        
        // 测试预分配版本
        float time_preallocated = testRadixSelectTopKPreallocatedThrust(h_input, h_output, h_indices, 
                                                              d_input, d_output, d_indices, 
                                                              h_reference_pairs, num_items, k, run, 
                                                              validate && (run % 10 == 0));
        total_time_preallocated += time_preallocated;
        
        if (run % 10 == 0) {
            printf("运行 %d: radixSelectTopK 用时: %.3f ms, radixSelectTopKPreallocated 用时: %.3f ms\n", 
                   run+1, time_standard, time_preallocated);
        }
    }
    
    // 输出平均性能结果
    float avg_time_standard = total_time_standard / num_runs;
    float avg_time_preallocated = total_time_preallocated / num_runs;
    
    printf("\n性能总结 (%d 次运行):\n", num_runs);
    printf("radixSelectTopKThrust 平均用时: %.3f ms\n", avg_time_standard);
    printf("radixSelectTopKPreallocatedThrust 平均用时: %.3f ms\n", avg_time_preallocated);
    printf("性能提升: %.2f%%\n", 100.0f * (avg_time_standard - avg_time_preallocated) / avg_time_standard);

    // 释放内存
    delete[] h_input;
    delete[] h_output;
    delete[] h_indices;
    delete[] h_reference_pairs;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);

    return 0;
}