#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// 强制编译器不内联，以便更准确地测量函数执行时间
#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE void ScalarAdd(int n, const float* a, const float* b, float* c) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

NOINLINE void AVX2Add(int n, const float* a, const float* b, float* c) {
int i = 0;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        
        // 增加算术强度：在不访问内存的情况下连续计算
        // 这一步模拟了 100 次 FMA 运算
        for(int j = 0; j < 1; j++) {
            va = _mm256_add_ps(va, vb);
            vb = _mm256_mul_ps(va, vb);
        }

        _mm256_storeu_ps(&c[i], va);
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 同样的逻辑：标量版本
NOINLINE void ScalarHighIntensity(int n, const float* a, const float* b, float* c) {
    for (int i = 0; i < n; i++) {
        float va = a[i];
        float vb = b[i];
        // 模拟高强度计算：100次迭代
        for(int j = 0; j < 100; j++) {
            va = va + vb;
            va = va * 0.999f; // 防止数值溢出成 infinity
        }
        c[i] = va;
    }
}

// 同样的逻辑：AVX2版本
NOINLINE void AVX2HighIntensity(int n, const float* a, const float* b, float* c) {
    int i = 0;
    __m256 v_factor = _mm256_set1_ps(0.999f);
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        
        for(int j = 0; j < 100; j++) {
            va = _mm256_add_ps(va, vb);
            va = _mm256_mul_ps(va, v_factor);
        }
        _mm256_storeu_ps(&c[i], va);
    }
    for (; i < n; i++) { /* 处理余数... */ }
}

// 优化后的高强度计算逻辑
NOINLINE void AVX2Optimized(int n, const float* a, const float* b, float* c) {
    __m256 v_f = _mm256_set1_ps(0.999f);
    int i = 0;
    for (i = 0; i <= n - 32; i += 32) {
        __m256 va1 = _mm256_loadu_ps(&a[i]);
        __m256 vb1 = _mm256_loadu_ps(&b[i]);
        __m256 va2 = _mm256_loadu_ps(&a[i+8]);
        __m256 vb2 = _mm256_loadu_ps(&b[i+8]);
        __m256 va3 = _mm256_loadu_ps(&a[i+16]);
        __m256 vb3 = _mm256_loadu_ps(&b[i+16]);
        __m256 va4 = _mm256_loadu_ps(&a[i+24]);
        __m256 vb4 = _mm256_loadu_ps(&b[i+24]);

        for(int j = 0; j < 100; j++) {
            va1 = _mm256_mul_ps(_mm256_add_ps(va1, vb1), v_f);
            va2 = _mm256_mul_ps(_mm256_add_ps(va2, vb2), v_f);
            va3 = _mm256_mul_ps(_mm256_add_ps(va3, vb3), v_f);
            va4 = _mm256_mul_ps(_mm256_add_ps(va4, vb4), v_f);
        }
        _mm256_storeu_ps(&c[i], va1);
        _mm256_storeu_ps(&c[i+8], va2);
        _mm256_storeu_ps(&c[i+16], va3);
        _mm256_storeu_ps(&c[i+24], va4);
    }
    // 处理剩余数据
    for (; i < n; i++) {
        float va = a[i];
        float vb = b[i];
        for(int j = 0; j < 100; j++) {
            va = (va + vb) * 0.999f;
        }
        c[i] = va;
    }
}


NOINLINE void AVX2Optimized8Way(int n, const float* a, const float* b, float* c) {
    __m256 v_f = _mm256_set1_ps(0.999f);
    int i = 0;
    // 每次处理 64 个元素
    for (i = 0; i <= n - 64; i += 64) {
        // 加载 8 路数据
        __m256 va1 = _mm256_loadu_ps(&a[i]);    __m256 vb1 = _mm256_loadu_ps(&b[i]);
        __m256 va2 = _mm256_loadu_ps(&a[i+8]);  __m256 vb2 = _mm256_loadu_ps(&b[i+8]);
        __m256 va3 = _mm256_loadu_ps(&a[i+16]); __m256 vb3 = _mm256_loadu_ps(&b[i+16]);
        __m256 va4 = _mm256_loadu_ps(&a[i+24]); __m256 vb4 = _mm256_loadu_ps(&b[i+24]);
        __m256 va5 = _mm256_loadu_ps(&a[i+32]); __m256 vb5 = _mm256_loadu_ps(&b[i+32]);
        __m256 va6 = _mm256_loadu_ps(&a[i+40]); __m256 vb6 = _mm256_loadu_ps(&b[i+40]);
        __m256 va7 = _mm256_loadu_ps(&a[i+48]); __m256 vb7 = _mm256_loadu_ps(&b[i+48]);
        __m256 va8 = _mm256_loadu_ps(&a[i+56]); __m256 vb8 = _mm256_loadu_ps(&b[i+56]);

        for(int j = 0; j < 100; j++) {
            // 这 8 行指令在 CPU 后端是几乎并行执行的
            va1 = _mm256_mul_ps(_mm256_add_ps(va1, vb1), v_f);
            va2 = _mm256_mul_ps(_mm256_add_ps(va2, vb2), v_f);
            va3 = _mm256_mul_ps(_mm256_add_ps(va3, vb3), v_f);
            va4 = _mm256_mul_ps(_mm256_add_ps(va4, vb4), v_f);
            va5 = _mm256_mul_ps(_mm256_add_ps(va5, vb5), v_f);
            va6 = _mm256_mul_ps(_mm256_add_ps(va6, vb6), v_f);
            va7 = _mm256_mul_ps(_mm256_add_ps(va7, vb7), v_f);
            va8 = _mm256_mul_ps(_mm256_add_ps(va8, vb8), v_f);
        }

        _mm256_storeu_ps(&c[i],    va1);
        _mm256_storeu_ps(&c[i+8],  va2);
        _mm256_storeu_ps(&c[i+16], va3);
        _mm256_storeu_ps(&c[i+24], va4);
        _mm256_storeu_ps(&c[i+32], va5);
        _mm256_storeu_ps(&c[i+40], va6);
        _mm256_storeu_ps(&c[i+48], va7);
        _mm256_storeu_ps(&c[i+56], va8);
    }
    // 处理剩余数据（标量版保持不变）
    for (; i < n; i++) {
        float va = a[i]; float vb = b[i];
        for(int j = 0; j < 100; j++) { va = (va + vb) * 0.999f; }
        c[i] = va;
    }
}

int main() {
    const int N = 20000000; // 2000万个元素
    const int ITERATIONS = 100; // 重复100次取平均值

    // 使用对齐的内存分配，虽然loadu支持不对齐，但对齐是最佳实践
    float* A = (float*)_mm_malloc(N * sizeof(float), 32);
    float* B = (float*)_mm_malloc(N * sizeof(float), 32);
    float* C_Scalar = (float*)_mm_malloc(N * sizeof(float), 32);
    float* C_AVX = (float*)_mm_malloc(N * sizeof(float), 32);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    cout << "Data Size: " << N * sizeof(float) / (1024 * 1024) << " MB" << endl;
    cout << "Running tests " << ITERATIONS << " times..." << endl;

    // --- 测试 Scalar 版本 ---
    auto start = high_resolution_clock::now();
    // for (int i = 0; i < ITERATIONS; i++) {
        // ScalarHighIntensity(N, A, B, C_Scalar);
    // }
    auto end = high_resolution_clock::now();
    // double scalar_time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;

    // --- 测试 AVX2 版本 ---
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        ScalarHighIntensity(N, A, B, C_AVX);
    }
    end = high_resolution_clock::now();
    double avx_time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;

    // 验证结果是否正确
    // for (int i = 0; i < N; i++) {
    //     if (abs(C_Scalar[i] - C_AVX[i]) > 1e-5) {
    //         cout << "Validation Failed at index " << i << endl;
    //         return -1;
    //     }
    // }

    // 输出结果
    cout << fixed << setprecision(2);
    cout << "--------------------------------------" << endl;
    // cout << "Scalar Average Time: " << scalar_time << " us" << endl;
    cout << "AVX2 Average Time:   " << avx_time << " us" << endl;
    // cout << "Speedup:             " << (scalar_time / avx_time) << "x" << endl;
    cout << "--------------------------------------" << endl;

    _mm_free(A);
    _mm_free(B);
    _mm_free(C_Scalar);
    _mm_free(C_AVX);

    return 0;
}
