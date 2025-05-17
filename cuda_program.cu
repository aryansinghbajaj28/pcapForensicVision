%%writefile cuda_program.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.14159265
#define MAX_KERNEL_SIZE 15

typedef unsigned short DATATYPE;

struct PGMstructure {
    int maxVal; //the maximum pixel value
    int width;//the dimensions of the image
    int height;
    DATATYPE *data; // a pointer to the image pixel data
};

// CUDA Kernel for Image Rotation
__global__ void img_rotate(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, float sinTheta, float cosTheta) {
    int ix = threadIdx.x;//x coordinate giving the column index
    int iy = blockIdx.x;//y coordinate giving the row index
    float x0 = W / 2.0f;
    float y0 = H / 2.0f;//calculating the center of image which is used as rotation origin
    float xOff = ix - x0;
    float yOff = iy - y0;//finding the position of the image relative to the center
    //applying the inverse rotation transformation to avoid the gaps in the output
    int xpos = (int)(xOff * cosTheta + yOff * sinTheta + x0);
    int ypos = (int)(yOff * cosTheta - xOff * sinTheta + y0);
 //checking if the computed new x and y components are within the bound of the input image
    if (xpos >= 0 && xpos < W && ypos >= 0 && ypos < H) {
        dest_data[iy * W + ix] = src_data[ypos * W + xpos];
    } else {
        dest_data[iy * W + ix] = 0; // Set out-of-bounds pixels to black
    }
}

// CUDA Kernel for Convolution
__global__ void convolution(DATATYPE *dest_data, DATATYPE *src_data, float *kernel, 
                            int W, int H, int kernelSize) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix < W && iy < H) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int x = ix + kx;
                int y = iy + ky;
                
                // Check boundaries
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    sum += src_data[y * W + x] * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                }
            }
        }
        
        // Clamp the result to the valid range
        if (sum < 0) sum = 0;
        if (sum > 65535) sum = 65535;  // Assuming 16-bit max value for DATATYPE
        
        dest_data[iy * W + ix] = (DATATYPE)sum;
    }
}

// CUDA Kernel for Median Filtering- replacing each pixel with the median of neighbouring pixels to remove the salt and pepper noise
__global__ void medianFilter(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, int windowSize) {
    //through this code snippet we calcualate the global pixel position
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //checking if the coordinate's are within the image bounds
    if (ix < W && iy < H) {
        int halfWindow = windowSize / 2;//how far can we go on each side from the fixel
        DATATYPE values[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE]; //array for storing the neighbour values
        int count = 0;//keeping the track of the values added
        
        //looping through the neighbourhood
        for (int ky = -halfWindow; ky <= halfWindow; ky++) {
            for (int kx = -halfWindow; kx <= halfWindow; kx++) {
                //computing x,y for each neighbour
                int x = ix + kx;
                int y = iy + ky;
                //adding the pixel if it is image bound
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    values[count++] = src_data[y * W + x];
                }
            }
        }
        
        // Simple bubble sort for sorting the collected data , suitable for smaller data size
        for (int i = 0; i < count-1; i++) {
            for (int j = 0; j < count-i-1; j++) {
                if (values[j] > values[j+1]) {
                    DATATYPE temp = values[j];
                    values[j] = values[j+1];
                    values[j+1] = temp;
                }
            }
        }
        
        // setting the median value in the destination
        dest_data[iy * W + ix] = values[count / 2];
    }
}

// CUDA Kernel for Morphological Dilation
__global__ void dilate(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, int kernelSize) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix < W && iy < H) {
        DATATYPE maxVal = 0;
        int halfKernel = kernelSize / 2;
        
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int x = ix + kx;
                int y = iy + ky;
                
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    if (src_data[y * W + x] > maxVal) {
                        maxVal = src_data[y * W + x];
                    }
                }
            }
        }
        
        dest_data[iy * W + ix] = maxVal;
    }
}

// CUDA Kernel for Morphological Erosion-processes each pixel of the image and replaces it with minimum value of the neighbourhood,it is used to shrink the brighter region and to remove the noise
__global__ void erode(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, int kernelSize) {
    //calculating the global pixel x and y coordinates
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    //ensuring the pixel is in the bound
    if (ix < W && iy < H) {
        DATATYPE minVal = 65535; // initialing the min_value
        int halfKernel = kernelSize / 2;//helps in defining the rangee of neighbourhood
        
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                //coordinates of the neighbouring pixel
                int x = ix + kx;
                int y = iy + ky;
                //ensuring the neighbouring pixels are within the bound
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    if (src_data[y * W + x] < minVal) {
                        minVal = src_data[y * W + x];//updating the min value
                    }
                }
            }
        }
        
        dest_data[iy * W + ix] = minVal;//assigning the smallest pixel value from the neighbourhood
    }
}

// Function to create Gaussian kernel
void createGaussianKernel(float *kernel, int kernelSize, float sigma) {
    int halfKernel = kernelSize / 2;
    float sum = 0.0f;
    
    for (int y = -halfKernel; y <= halfKernel; y++) {
        for (int x = -halfKernel; x <= halfKernel; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + halfKernel) * kernelSize + (x + halfKernel)] = value;
            sum += value;
        }
    }
    
    // Normalize the kernel
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
}

// Function to create edge detection (Sobel) kernel - this function helps in detection of the images specially horizontal and vertical gradients in an image
void createSobelKernel(float *kernelX, float *kernelY, int kernelSize) {
    // Horizontal Sobel kernel storing a 3*3 matrix row by row into flat 1d array it helpsi id detecting changes in horizontal direction which helps to find the edges along X axis intensity changes from left to right
    kernelX[0] = -1.0f; kernelX[1] = 0.0f; kernelX[2] = 1.0f;
    kernelX[3] = -2.0f; kernelX[4] = 0.0f; kernelX[5] = 2.0f;
    kernelX[6] = -1.0f; kernelX[7] = 0.0f; kernelX[8] = 1.0f;
    
    // Vertical Sobel kernel storing a 3*3 kernel which stores value column to column help in detecting vertical edges , intensity changes from top to bottom
    kernelY[0] = -1.0f; kernelY[1] = -2.0f; kernelY[2] = -1.0f;
    kernelY[3] = 0.0f;  kernelY[4] = 0.0f;  kernelY[5] = 0.0f;
    kernelY[6] = 1.0f;  kernelY[7] = 2.0f;  kernelY[8] = 1.0f;
}

// Function to read PGM file
struct PGMstructure* readPGM(const char* filename) {
    FILE *imagein;
    int row, col;
    unsigned int ch_int;

    struct PGMstructure *imginfo = (struct PGMstructure *)malloc(sizeof(struct PGMstructure));
    
    imagein = fopen(filename, "r");
    if (imagein == NULL) {
        printf("Error opening file %s\n", filename);
        free(imginfo);
        return NULL;
    }

    char magicNumber[3];
    fscanf(imagein, "%s", magicNumber);
    fscanf(imagein, "%d %d", &imginfo->width, &imginfo->height);
    fscanf(imagein, "%d", &imginfo->maxVal);

    imginfo->data = (DATATYPE *)malloc(imginfo->width * imginfo->height * sizeof(DATATYPE));

    for (row = 0; row < imginfo->height; row++) {
        for (col = 0; col < imginfo->width; col++) {
            fscanf(imagein, "%u", &ch_int);
            imginfo->data[row * imginfo->width + col] = ch_int;
        }
    }
    fclose(imagein);
    
    return imginfo;
}

// Function to write PGM file
void writePGM(const char* filename, struct PGMstructure* img, DATATYPE* data) {
    FILE *imageout;//filepointer to be used to write in the file
    int row, col;
    
    imageout = fopen(filename, "w");
    if (imageout == NULL) {
        printf("Error opening output file %s\n", filename);
        return;
    }
    //writing the head of the image file
    fprintf(imageout, "P2\n%d %d\n%d\n", img->width, img->height, img->maxVal);
    //nested loop writing each value row by row
    for (row = 0; row < img->height; row++) {
        for (col = 0; col < img->width; col++) {
            fprintf(imageout, "%d ", data[row * img->width + col]);
        }
        fprintf(imageout, "\n");
    }
    
    fclose(imageout);
}

int main() {
    char inputPath[1000], outputPath[1000];
    int choice, parameter;
    
    // Get input file
    printf("Enter PGM file path: ");
    scanf("%s", inputPath);
    
    // Read input image
    struct PGMstructure *imginfo = readPGM(inputPath);
    if (imginfo == NULL) {
        return -1;
    }
    
    // Display menu
    printf("\nForensicVision: GPU-Accelerated Image Enhancement\n");
    printf("1. Image Rotation\n");
    printf("2. Gaussian Blur\n");
    printf("3. Edge Detection (Sobel)\n");
    printf("4. Median Filter (Noise Reduction)\n");
    printf("5. Morphological Dilation\n");
    printf("6. Morphological Erosion\n");
    printf("Choose operation (1-6): ");
    scanf("%d", &choice);
    
    // Allocate device memory
    DATATYPE *d_src, *d_dst, *d_temp, *dst;
    int size = imginfo->width * imginfo->height * sizeof(DATATYPE);
    
    cudaMalloc((void **)&d_src, size);
    cudaMalloc((void **)&d_dst, size);
    cudaMemcpy(d_src, imginfo->data, size, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time = 0.0f;
    
    // Start timing
    cudaEventRecord(start, 0);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error before processing: %s\n", cudaGetErrorString(error));
    }
    
    // Processing based on user choice
    switch (choice) {
        case 1: { // Image Rotation
            int angle;
            printf("Enter angle to rotate (degrees): ");
            scanf("%d", &angle);
            
            float sinTheta = sinf(angle * PI / 180.0f);
            float cosTheta = cosf(angle * PI / 180.0f);
            
            img_rotate<<<imginfo->height, imginfo->width>>>(d_dst, d_src, 
                                                           imginfo->width, imginfo->height, 
                                                           sinTheta, cosTheta);
            break;
        }
        
        case 2: { // Gaussian Blur
            int kernelSize;
            float sigma;
            
            printf("Enter kernel size (odd number, e.g., 3, 5, 7): ");
            scanf("%d", &kernelSize);
            printf("Enter sigma value (e.g., 1.0): ");
            scanf("%f", &sigma);
            
            // Create Gaussian kernel
            float *h_kernel = (float *)malloc(kernelSize * kernelSize * sizeof(float));
            createGaussianKernel(h_kernel, kernelSize, sigma);
            
            // Copy kernel to device
            float *d_kernel;
            cudaMalloc((void **)&d_kernel, kernelSize * kernelSize * sizeof(float));
            cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
            
            // Define thread block and grid dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((imginfo->width + blockDim.x - 1) / blockDim.x, 
                         (imginfo->height + blockDim.y - 1) / blockDim.y);
            
            // Launch kernel
            convolution<<<gridDim, blockDim>>>(d_dst, d_src, d_kernel, 
                                             imginfo->width, imginfo->height, kernelSize);
            
            // Clean up
            cudaFree(d_kernel);
            free(h_kernel);
            break;
        }
        
        case 3: { // Edge Detection (Sobel)
            // Create Sobel kernels (X and Y directions)
            float h_kernelX[9], h_kernelY[9];
            createSobelKernel(h_kernelX, h_kernelY, 3);
            
            // Copy kernels to device
            float *d_kernelX, *d_kernelY;
            cudaMalloc((void **)&d_kernelX, 9 * sizeof(float));
            cudaMalloc((void **)&d_kernelY, 9 * sizeof(float));
            cudaMemcpy(d_kernelX, h_kernelX, 9 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernelY, h_kernelY, 9 * sizeof(float), cudaMemcpyHostToDevice);
            
            // Allocate temporary buffer for Y direction gradient
            cudaMalloc((void **)&d_temp, size);
            
            // Define thread block and grid dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((imginfo->width + blockDim.x - 1) / blockDim.x, 
                         (imginfo->height + blockDim.y - 1) / blockDim.y);
            
            // Apply X direction Sobel
            convolution<<<gridDim, blockDim>>>(d_dst, d_src, d_kernelX, 
                                             imginfo->width, imginfo->height, 3);
            
            // Apply Y direction Sobel
            convolution<<<gridDim, blockDim>>>(d_temp, d_src, d_kernelY, 
                                             imginfo->width, imginfo->height, 3);
            
            // Clean up
            cudaFree(d_kernelX);
            cudaFree(d_kernelY);
            cudaFree(d_temp);
            break;
        }
        
        case 4: { // Median Filter
            int windowSize;
            printf("Enter window size (odd number, e.g., 3, 5): ");
            scanf("%d", &windowSize);
            
            if (windowSize > MAX_KERNEL_SIZE) {
                printf("Window size too large, using maximum: %d\n", MAX_KERNEL_SIZE);
                windowSize = MAX_KERNEL_SIZE;
            }
            
            // Define thread block and grid dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((imginfo->width + blockDim.x - 1) / blockDim.x, 
                         (imginfo->height + blockDim.y - 1) / blockDim.y);
            
            // Launch kernel
            medianFilter<<<gridDim, blockDim>>>(d_dst, d_src, 
                                              imginfo->width, imginfo->height, windowSize);
            break;
        }
        
        case 5: { // Morphological Dilation
            int kernelSize;
            printf("Enter structuring element size (odd number, e.g., 3, 5): ");
            scanf("%d", &kernelSize);
            
            // Define thread block and grid dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((imginfo->width + blockDim.x - 1) / blockDim.x, 
                         (imginfo->height + blockDim.y - 1) / blockDim.y);
            
            // Launch kernel
            dilate<<<gridDim, blockDim>>>(d_dst, d_src, 
                                        imginfo->width, imginfo->height, kernelSize);
            break;
        }
        
        case 6: { // Morphological Erosion
            int kernelSize;
            printf("Enter structuring element size (odd number, e.g., 3, 5): ");
            scanf("%d", &kernelSize);
            
            // Define thread block and grid dimensions
            dim3 blockDim(16, 16);
            dim3 gridDim((imginfo->width + blockDim.x - 1) / blockDim.x, 
                         (imginfo->height + blockDim.y - 1) / blockDim.y);
            
            // Launch kernel
            erode<<<gridDim, blockDim>>>(d_dst, d_src, 
                                       imginfo->width, imginfo->height, kernelSize);
            break;
        }
        
        default:
            printf("Invalid choice\n");
            cudaFree(d_src);
            cudaFree(d_dst);
            free(imginfo->data);
            free(imginfo);
            return -1;
    }
    
    // Check for CUDA errors after kernel execution
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error during processing: %s\n", cudaGetErrorString(error));
    }
    
    // Stop timing and calculate elapsed time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Total GPU processing time: %f ms\n", elapsed_time);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Get output file path
    printf("Enter output file path: ");
    scanf("%s", outputPath);
    
    // Copy result back to host
    dst = (DATATYPE *)malloc(size);
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);
    
    // Write output
    writePGM(outputPath, imginfo, dst);
    
    // Clean up
    free(imginfo->data);
    free(imginfo);
    free(dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    
    printf("Processing complete. Output saved to %s\n", outputPath);
    
    return 0;
}