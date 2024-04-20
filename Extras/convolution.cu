#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void conv2d(const float* x,
            float* out,
            const float* conv_kernel,
            const int* x_shape,
            const int kernel_size,
            const int in_channels,
            const int out_channels,
            const int out_width,
            const int out_height
            ){
    float aggregator = 0.0;
    // current convolution doesn't support padding or stride which doesn't equal 1
    // just a basic implementation for a single image
    
    // Iterate over each output channel of the conv kernel
    for (int kernel_idx = 0; kernel_idx < out_channels; kernel_idx++){
        // Iterate over the width of the image
        for (int i = 0; i < x_shape[1] - kernel_size + 1; i++){
            // Iterate over the height of the image
            for (int j = 0; j < x_shape[0] - kernel_size + 1; j++){
                // Iterate over the chanenls of the image(calculation for a single output pixel of the conv output starts here)
                for (int c = 0; c < in_channels; c++){
                    // Iterate over the image width depending on the kernel size
                    for (int k = 0; k < kernel_size; k++){
                        // Iterate over the image height depending on the kernel size
                        for (int l = 0; l < kernel_size; l++){  
                            // Calculate indices for both the image and kernel
                            int x_index = (i + l) * in_channels + (j + k) * x_shape[1] * in_channels + c;
                            int kernel_index = kernel_idx * kernel_size * kernel_size * in_channels + l * in_channels + k * kernel_size * in_channels + c;
                            // aggregate the conv op output to a single aggregator
                            aggregator += x[x_index] * conv_kernel[kernel_index];
                        }
                    }
                }
            // Write the aggregator to the output pixel after it has iterated over all channels, for the whole convolution
            out[i + j * out_width + kernel_idx * out_width * out_height] = aggregator;
            aggregator = 0.0;
            }
        }
    }
};

// CUDA implementation of the above function
__global__
void conv2dKernel(
    const float* x,
    float* out,
    const float* conv_kernel,
    const int* x_shape,
    const int kernel_size,
    const int in_channels,
    const int out_channels,
    const int out_width,
    const int out_height
){
    float aggregator = 0.0;
    int kernel_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.z * blockIdx.z + threadIdx.z;

    if ((kernel_idx < out_channels) && (i < x_shape[1] - kernel_size + 1) && (j < x_shape[0] - kernel_size + 1)){
        float aggregator = 0.0;
        for (int c = 0; c < in_channels; c++){
            for (int k = 0; k < kernel_size; k++){
                for (int l = 0; l < kernel_size; l++){  
                    int x_index = (i + l) * in_channels + (j + k) * x_shape[1] * in_channels + c;
                    int kernel_index = kernel_idx * kernel_size * kernel_size * in_channels + l * in_channels + k * kernel_size * in_channels + c;
                    aggregator += x[x_index] * conv_kernel[kernel_index];
                }
            }
        }
        out[i + j * out_width + kernel_idx * out_width * out_height] = aggregator;
    }
};

float* init_out_feature(const int* x_shape, 
                        int conv_kernel_shape, 
                        int out_channels,
                        int& output_height,
                        int& output_width
                        ){
    // Initialize output feature from an image and 
    // write the height and width of it in the output_height and output_width
    // variables
    output_height = floor((x_shape[0] - conv_kernel_shape)) + 1;
    output_width = floor((x_shape[1] - conv_kernel_shape)) + 1;
    float* out = (float*)malloc(output_height * output_width * out_channels * sizeof(float));
    return out;
};

float* load_image(const char* image_path, int& width, int& height, int& channels){
    float *img = stbi_loadf(image_path, &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error loading image\n");
        exit(1);
    }
    return img;
};

void initConvKernel(float* kernel, int numel){
    for (int i = 0; i < numel; i++){
        kernel[i] = 1.0;
    }
};

void free_tensors(float* x, float* out, float* conv_kernel){
    stbi_image_free(x);
    free(out);
    free(conv_kernel);
};

void printImageValues(float* img){
    // Print image values C, H, W
    // to figure out the correct strides
    printf("%f\n", img[0]);
    printf("%f\n", img[1]);
    printf("%f\n", img[2]);
    printf("%f\n", img[3]);
    printf("%f\n", img[4]);
    printf("%f\n", img[5]);
};

void conv2dGPU(float* x, 
               float* out,
               float* conv_kernel, 
               int* shape, 
               const int output_height, 
               const int output_width, 
               const int kernel_size, 
               const int out_ch, 
               const int in_ch, 
               const int width, 
               const int height){
    // Allocations on GPU
    float* x_d, *out_d, *conv_kernel_d;
    int* shape_d;
    
    dim3 dimBlock(16, 16, 4);
    dim3 dimGrid((output_width + dimBlock.x - 1) / dimBlock.x, 
                 (output_height + dimBlock.y - 1) / dimBlock.y,
                 (out_ch + dimBlock.z - 1) / dimBlock.z
                 );
    // Allocate memory on device
    cudaMalloc((void**)&x_d, width * height * in_ch * sizeof(float));
    cudaMalloc((void**)&out_d, output_height * output_width * out_ch * sizeof(float));
    cudaMalloc((void**)&conv_kernel_d, kernel_size * kernel_size * out_ch * in_ch * sizeof(float));
    cudaMalloc((void**)&shape_d, 3 * sizeof(int));
    // Copy the tensors from Host to device
    cudaMemcpy(x_d, x, width * height * in_ch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(conv_kernel_d, conv_kernel, kernel_size * kernel_size * out_ch * in_ch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_d, shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    // Run the kernel
    conv2dKernel<<<dimGrid, dimBlock>>>(x_d, out_d, conv_kernel_d, shape_d, kernel_size, in_ch, out_ch, output_width, output_height);
    cudaMemcpy(out, out_d, output_height * output_width * out_ch * sizeof(float), cudaMemcpyDeviceToHost);
    // Free the tensors on device
    cudaFree(x_d);
    cudaFree(out_d);
    cudaFree(conv_kernel_d);
    cudaFree(shape_d);
}

int main(int argc, const char* argv[]){
    // Currently supports only square conv kernels
    int width, height, in_ch;
    float* x = load_image("some_img.jpg", width, height, in_ch);
    int shape[3] = {height, width, in_ch};
    int kernel_size = atoi(argv[1]);
    int out_ch = atoi(argv[2]);
    int numel = shape[0] * shape[1] * shape[2];
    int output_height, output_width;
    float* out = init_out_feature(shape, kernel_size, out_ch, output_height, output_width);
    float* conv_kernel = (float*)malloc(kernel_size * kernel_size * out_ch * in_ch * sizeof(float));
    initConvKernel(conv_kernel, kernel_size * kernel_size * out_ch * in_ch);

    // GPU Conv2d
    conv2dGPU(x, out, conv_kernel, shape, output_height, output_width, kernel_size, out_ch, in_ch, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // CPU Conv2d
    // conv2d(x, out, conv_kernel, shape, kernel_size, in_ch, out_ch, output_width, output_height);

    // printImageValues(out);
    free_tensors(x, out, conv_kernel);
}
