#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PI 3.14159265
#define MAX_KERNEL_SIZE 15

typedef unsigned short DATATYPE;

struct PGMstructure {
    int maxVal; // The maximum pixel value
    int width;  // The dimensions of the image
    int height;
    DATATYPE *data; // A pointer to the image pixel data
};

// Function to rotate a portion of an image
void img_rotate(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, 
                int start_row, int end_row, float sinTheta, float cosTheta) {
    float x0 = W / 2.0f;
    float y0 = H / 2.0f; 
    
    for (int iy = start_row; iy < end_row; iy++) {
        for (int ix = 0; ix < W; ix++) {
            float xOff = ix - x0;
            float yOff = iy - y0;
            int xpos = (int)(xOff * cosTheta + yOff * sinTheta + x0);
            int ypos = (int)(yOff * cosTheta - xOff * sinTheta + y0);
            
            if (xpos >= 0 && xpos < W && ypos >= 0 && ypos < H) {
                dest_data[(iy-start_row) * W + ix] = src_data[ypos * W + xpos];
            } else {
                dest_data[(iy-start_row) * W + ix] = 0;
            }
        }
    }
}

// Function for convolution operation
void convolution(DATATYPE *dest_data, DATATYPE *src_data, float *kernel, 
                int W, int H, int start_row, int end_row, int kernelSize) {
    int halfKernel = kernelSize / 2;
    
    for (int iy = start_row; iy < end_row; iy++) {
        for (int ix = 0; ix < W; ix++) {
            float sum = 0.0f;
            
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int x = ix + kx;
                    int y = iy + ky;
                    
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        sum += src_data[y * W + x] * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                    }
                }
            }
            
            if (sum < 0) sum = 0;
            if (sum > 65535) sum = 65535;
            
            dest_data[(iy-start_row) * W + ix] = (DATATYPE)sum;
        }
    }
}

// Function for median filtering
void medianFilter(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, 
                 int start_row, int end_row, int windowSize) {
    int halfWindow = windowSize / 2;
    
    for (int iy = start_row; iy < end_row; iy++) {
        for (int ix = 0; ix < W; ix++) {
            DATATYPE values[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
            int count = 0;
            
            for (int ky = -halfWindow; ky <= halfWindow; ky++) {
                for (int kx = -halfWindow; kx <= halfWindow; kx++) {
                    int x = ix + kx;
                    int y = iy + ky;
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        values[count++] = src_data[y * W + x];
                    }
                }
            }
            
            for (int i = 0; i < count-1; i++) {
                for (int j = 0; j < count-i-1; j++) {
                    if (values[j] > values[j+1]) {
                        DATATYPE temp = values[j];
                        values[j] = values[j+1];
                        values[j+1] = temp;
                    }
                }
            }
            
            dest_data[(iy-start_row) * W + ix] = values[count / 2];
        }
    }
}

// Function for morphological dilation
void dilate(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, 
           int start_row, int end_row, int kernelSize) {
    int halfKernel = kernelSize / 2;
    
    for (int iy = start_row; iy < end_row; iy++) {
        for (int ix = 0; ix < W; ix++) {
            DATATYPE maxVal = 0;
            
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
            
            dest_data[(iy-start_row) * W + ix] = maxVal;
        }
    }
}

// Function for morphological erosion
void erode(DATATYPE *dest_data, DATATYPE *src_data, int W, int H, 
          int start_row, int end_row, int kernelSize) {
    int halfKernel = kernelSize / 2;
    
    for (int iy = start_row; iy < end_row; iy++) {
        for (int ix = 0; ix < W; ix++) {
            DATATYPE minVal = 65535;
            
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int x = ix + kx;
                    int y = iy + ky;
                    
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        if (src_data[y * W + x] < minVal) {
                            minVal = src_data[y * W + x];
                        }
                    }
                }
            }
            
            dest_data[(iy-start_row) * W + ix] = minVal;
        }
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

// Function to create edge detection (Sobel) kernel
void createSobelKernel(float *kernelX, float *kernelY) {
    // Horizontal Sobel kernel
    kernelX[0] = -1.0f; kernelX[1] = 0.0f; kernelX[2] = 1.0f;
    kernelX[3] = -2.0f; kernelX[4] = 0.0f; kernelX[5] = 2.0f;
    kernelX[6] = -1.0f; kernelX[7] = 0.0f; kernelX[8] = 1.0f;
    
    // Vertical Sobel kernel
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
    FILE *imageout;
    int row, col;
    
    imageout = fopen(filename, "w");
    if (imageout == NULL) {
        printf("Error opening output file %s\n", filename);
        return;
    }
    
    fprintf(imageout, "P2\n%d %d\n%d\n", img->width, img->height, img->maxVal);
    
    for (row = 0; row < img->height; row++) {
        for (col = 0; col < img->width; col++) {
            fprintf(imageout, "%d ", data[row * img->width + col]);
        }
        fprintf(imageout, "\n");
    }
    
    fclose(imageout);
}

int main(int argc, char *argv[]) {
    int rank, size, choice;
    char inputPath[1000], outputPath[1000];
    struct PGMstructure *imginfo = NULL;
    DATATYPE *result_data = NULL;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Root process handles input
    if (rank == 0) {
        printf("Enter PGM file path: ");
        fflush(stdout);  // Add flush to ensure prompt is displayed
        scanf("%s", inputPath);
        
        // Read input image
        imginfo = readPGM(inputPath);
        if (imginfo == NULL) {
            printf("Failed to read image file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }
        
        // Display menu
        printf("\nForensicVision: MPI-Accelerated Image Enhancement\n");
        printf("1. Image Rotation\n");
        printf("2. Gaussian Blur\n");
        printf("3. Edge Detection (Sobel)\n");
        printf("4. Median Filter (Noise Reduction)\n");
        printf("5. Morphological Dilation\n");
        printf("6. Morphological Erosion\n");
        printf("Choose operation (1-6): ");
        fflush(stdout);  // Add flush to ensure prompt is displayed
        scanf("%d", &choice);
        
        // Allocate memory for result
        result_data = (DATATYPE *)malloc(imginfo->width * imginfo->height * sizeof(DATATYPE));
    }
    
    // Broadcast the choice to all processes
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // If not a valid choice, exit
    if (choice < 1 || choice > 6) {
        if (rank == 0) {
            printf("Invalid choice\n");
            if (imginfo) {
                free(imginfo->data);
                free(imginfo);
            }
            if (result_data) free(result_data);
        }
        MPI_Finalize();
        return -1;
    }
    
    // Broadcast the image info to all processes
    int dimensions[3];  // width, height, maxVal
    if (rank == 0) {
        dimensions[0] = imginfo->width;
        dimensions[1] = imginfo->height;
        dimensions[2] = imginfo->maxVal;
    }
    
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    int width = dimensions[0];
    int height = dimensions[1];
    int maxVal = dimensions[2];
    
    // Each process needs to know the size of the image
    int image_size = width * height;
    
    // Allocate memory for source data for all processes
    DATATYPE *src_data = NULL;
    
    if (rank == 0) {
        src_data = imginfo->data;
    } else {
        src_data = (DATATYPE *)malloc(image_size * sizeof(DATATYPE));
    }
    
    // Broadcast the entire image data
    MPI_Bcast(src_data, image_size, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
    
    // Calculate the portion of rows each process will handle
    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_process;
    int local_rows = end_row - start_row;
    
    // Allocate memory for local result
    DATATYPE *local_result = (DATATYPE *)malloc(width * local_rows * sizeof(DATATYPE));
    
    // Start timing
    start_time = MPI_Wtime();
    
    // Processing based on user choice
    switch (choice) {
        case 1: { // Image Rotation
            int angle;
            if (rank == 0) {
                printf("Enter angle to rotate (degrees): ");
                fflush(stdout);
                scanf("%d", &angle);
            }
            
            // Broadcast angle to all processes
            MPI_Bcast(&angle, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            float sinTheta = sinf(angle * PI / 180.0f);
            float cosTheta = cosf(angle * PI / 180.0f);
            
            // Allocate memory for full image result
            DATATYPE *temp_result = NULL;
            if (rank == 0) {
                temp_result = (DATATYPE *)calloc(image_size, sizeof(DATATYPE));
            }
            
            // Each process processes its portion
            img_rotate(local_result, src_data, width, height, start_row, end_row, sinTheta, cosTheta);
            
            // Gather results back to root process
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT, 
                      (rank == 0) ? temp_result + start_row * width : NULL, 
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                memcpy(result_data, temp_result, image_size * sizeof(DATATYPE));
                free(temp_result);
            }
            break;
        }
        
        case 2: { // Gaussian Blur
            int kernelSize;
            float sigma;
            
            if (rank == 0) {
                printf("Enter kernel size (odd number, e.g., 3, 5, 7): ");
                fflush(stdout);
                scanf("%d", &kernelSize);
                printf("Enter sigma value (e.g., 1.0): ");
                fflush(stdout);
                scanf("%f", &sigma);
            }
            
            // Broadcast parameters to all processes
            MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&sigma, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            
            // Create Gaussian kernel
            float *kernel = (float *)malloc(kernelSize * kernelSize * sizeof(float));
            createGaussianKernel(kernel, kernelSize, sigma);
            
            // Each process performs convolution on its portion
            convolution(local_result, src_data, kernel, width, height, start_row, end_row, kernelSize);
            
            // Gather results from all processes
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT,
                      (rank == 0) ? result_data : NULL,
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            free(kernel);
            break;
        }
        
        case 3: { // Edge Detection (Sobel)
            // Create Sobel kernels (X and Y directions)
            float kernelX[9], kernelY[9];
            createSobelKernel(kernelX, kernelY);
            
            // Each process performs convolution on its portion for X and Y directions
            DATATYPE *local_x = (DATATYPE *)malloc(width * local_rows * sizeof(DATATYPE));
            DATATYPE *local_y = (DATATYPE *)malloc(width * local_rows * sizeof(DATATYPE));
            
            // Apply X and Y direction Sobel filters
            convolution(local_x, src_data, kernelX, width, height, start_row, end_row, 3);
            convolution(local_y, src_data, kernelY, width, height, start_row, end_row, 3);
            
            // Combine the results using the gradient magnitude
            for (int i = 0; i < width * local_rows; i++) {
                int gx = local_x[i];
                int gy = local_y[i];
                local_result[i] = (DATATYPE)sqrt(gx*gx + gy*gy);
                if (local_result[i] > 65535) local_result[i] = 65535;
            }
            
            // Gather results from all processes
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT,
                      (rank == 0) ? result_data : NULL,
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            free(local_x);
            free(local_y);
            break;
        }
        
        case 4: { // Median Filter
            int windowSize;
            
            if (rank == 0) {
                printf("Enter window size (odd number, e.g., 3, 5): ");
                fflush(stdout);
                scanf("%d", &windowSize);
                
                if (windowSize > MAX_KERNEL_SIZE) {
                    printf("Window size too large, using maximum: %d\n", MAX_KERNEL_SIZE);
                    windowSize = MAX_KERNEL_SIZE;
                }
            }
            
            // Broadcast parameters to all processes
            MPI_Bcast(&windowSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Each process performs median filtering on its portion
            medianFilter(local_result, src_data, width, height, start_row, end_row, windowSize);
            
            // Gather results from all processes
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT,
                      (rank == 0) ? result_data : NULL,
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            break;
        }
        
        case 5: { // Morphological Dilation
            int kernelSize;
            
            if (rank == 0) {
                printf("Enter structuring element size (odd number, e.g., 3, 5): ");
                fflush(stdout);
                scanf("%d", &kernelSize);
            }
            
            // Broadcast parameters to all processes
            MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Each process performs dilation on its portion
            dilate(local_result, src_data, width, height, start_row, end_row, kernelSize);
            
            // Gather results from all processes
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT,
                      (rank == 0) ? result_data : NULL,
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            break;
        }
        
        case 6: { // Morphological Erosion
            int kernelSize;
            
            if (rank == 0) {
                printf("Enter structuring element size (odd number, e.g., 3, 5): ");
                fflush(stdout);
                scanf("%d", &kernelSize);
            }
            
            // Broadcast parameters to all processes
            MPI_Bcast(&kernelSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Each process performs erosion on its portion
            erode(local_result, src_data, width, height, start_row, end_row, kernelSize);
            
            // Gather results from all processes
            MPI_Gather(local_result, width * local_rows, MPI_UNSIGNED_SHORT,
                      (rank == 0) ? result_data : NULL,
                      width * local_rows, MPI_UNSIGNED_SHORT, 0, MPI_COMM_WORLD);
            
            break;
        }
    }
    
    // End timing
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Total processing time: %f seconds\n", end_time - start_time);
        
        // Get output file path
        printf("Enter output file path: ");
        fflush(stdout);
        scanf("%s", outputPath);
        
        // Create a PGM structure for the output image
        struct PGMstructure outimg;
        outimg.width = width;
        outimg.height = height;
        outimg.maxVal = maxVal;
        
        // Write output
        writePGM(outputPath, &outimg, result_data);
        
        printf("Processing complete. Output saved to %s\n", outputPath);
        
        // Clean up
        free(imginfo->data);
        free(imginfo);
        free(result_data);
    }
    
    // Clean up for all processes
    if (rank != 0) {
        free(src_data);
    }
    free(local_result);
    
    MPI_Finalize();
    return 0;
}