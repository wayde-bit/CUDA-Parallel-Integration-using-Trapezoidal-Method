#include <cuda.h>
#include <stdio.h>
#include <math.h>

__device__ double f(double x) {
    return x * x;
}

__global__ void trapezoidalKernel(double a, double h, int n, double *partial_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n-1) {
        double x = a + idx * h;
        double fx = f(x);
        double fx_next = f(x + h);
        partial_sum[idx] = h * (fx + fx_next) / 2.0;
    }
}

double cudaIntegrate(double a, double b, int n) {
    double h = (b - a) / n; 
    double *d_partial_sum;
    cudaMalloc(&d_partial_sum, (n-1) * sizeof(double));
    
    int blockSize = 256;
    int gridSize = (n-1 + blockSize - 1) / blockSize;
    
    trapezoidalKernel<<<gridSize, blockSize>>>(a, h, n, d_partial_sum);  
    cudaDeviceSynchronize();
    
    double *h_partial_sum = (double*)malloc((n-1) * sizeof(double));
    cudaMemcpy(h_partial_sum, d_partial_sum, (n-1) * sizeof(double), cudaMemcpyDeviceToHost);
    
    double sum = 0.0;
    for (int i = 0; i < n-1; i++) {
        sum += h_partial_sum[i];
    }
    
    cudaFree(d_partial_sum);
    free(h_partial_sum);
    
    return sum;
}

int main() {
    printf("CUDA Parallel Integration using Trapezoidal Method\n");
    
    // Test Case 1
    double a = 0.0, b = 1.0;
    int n = 1000;
    
    printf("Test Case 1:\n");
    printf("Function: f(x) = x^2\n");
    printf("Interval: [%.1f, %.1f]\n", a, b);
    printf("Number of intervals: %d\n", n);
    
    double analytical = (b*b*b - a*a*a) / 3.0;
    printf("Analytical result: %.10f\n", analytical);
    
    double cuda_result = cudaIntegrate(a, b, n);
    printf("CUDA result: %.10f\n", cuda_result);
    printf("Error: %.10f\n\n", fabs(cuda_result - analytical));
    
    // Test Case 2
    a = 1.0; b = 3.0; n = 5000;
    
    printf("Test Case 2:\n");
    printf("Function: f(x) = x^2\n");
    printf("Interval: [%.1f, %.1f]\n", a, b);
    printf("Number of intervals: %d\n", n);
    
    analytical = (b*b*b - a*a*a) / 3.0;
    printf("Analytical result: %.10f\n", analytical);
    
    cuda_result = cudaIntegrate(a, b, n);
    printf("CUDA result: %.10f\n", cuda_result);
    printf("Error: %.10f\n", fabs(cuda_result - analytical));
    
    return 0;
}
