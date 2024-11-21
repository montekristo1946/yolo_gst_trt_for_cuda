#include <iostream>
#include <cuda_runtime.h>

//
__global__ void multiplyArrays(float *dev_a, float *dev_b, float *dev_c, int N) {
    float count = dev_a[0];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        dev_c[index] = dev_a[index] + dev_b[index];
    }
}


int main() {
    int N = 100;

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f * i / N;
        h_b[i] = 1.0f * i / N;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));


    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    multiplyArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);


    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }


    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++) {
        std::cout << "h_c[" << i << "] = " << h_c[i] << std::endl;
    }


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}