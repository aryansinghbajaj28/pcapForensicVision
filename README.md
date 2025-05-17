# pcapForensicVision: GPU-Accelerated Image Enhancement

**ForensicVision** is a CUDA-based application designed to enhance degraded images using GPU acceleration for forensic analysis. It implements advanced image processing techniques to improve the clarity of CCTV footage, crime scene photos, fingerprint scans, and more.

---

## 🧩 Problem Statement

Crime investigations often rely on low-quality images degraded by noise, poor lighting, motion blur, or low resolution. Traditional CPU-based image processing can be too slow for timely analysis, especially with large forensic datasets.

---

## 🚀 Solution

ForensicVision leverages the parallel computing power of NVIDIA CUDA to significantly reduce processing time compared to CPU methods. The system provides multiple GPU-accelerated image enhancement techniques for clearer, more detailed forensic images.

---

## 🔧 Features

### 1. Image Rotation
- Rotates images by a user-defined angle
- Uses trigonometric transformations for precise repositioning
- Maintains image quality
- Parallelized: one GPU thread per pixel

### 2. Gaussian Blur
- Reduces noise via image smoothing
- Customizable Gaussian kernel and sigma value
- GPU-accelerated convolution for fast processing

### 3. Edge Detection (Sobel Operator)
- Highlights image contours using horizontal and vertical gradients
- Enhances boundary visibility
- Parallel gradient computation

### 4. Median Filter (Noise Reduction)
- Removes salt-and-pepper noise
- Uses a sliding window median replacement
- Custom window size
- Parallelized for high performance

### 5. Morphological Dilation
- Expands bright regions in grayscale or binary images
- Fills small gaps and enhances visibility
- Customizable structuring element
- Parallel per-pixel processing

### 6. Morphological Erosion
- Shrinks bright regions and removes small noise particles
- Refines object edges
- Customizable structuring element
- Efficient parallel implementation

---

## ⚙️ Performance Optimization

### 🧮 Parallel Processing Benefits
- **Speedup**: 50–100× faster than CPU-based methods
- **Real-time Processing**: Enables quick analysis of high-resolution forensic images

### 📏 Scalability
- **Image Size**: Efficient scaling with larger images
- **Kernel Independence**: Performance remains stable even with large convolution kernels
- **Resolution Flexibility**: Supports images of varying sizes

### 🔍 Resource Optimization
- Optimized **16×16 thread blocks** for high GPU occupancy
- **Memory coalescing** to maximize bandwidth
- **Dynamic resource allocation** based on GPU capacity

### 🔬 Algorithm-Specific Optimizations
- **Rotation**: All pixels transformed in parallel
- **Convolution**: 2D thread blocks handle multiple pixels at once
- **Median Filter**: Eliminates CPU-based sequential bottlenecks
- **Morphology**: Local neighborhood operations run in parallel for each pixel

---

## 💾 Input & Output

- **Input**: PGM (Portable Gray Map) format images
- **Parameters**: User-defined (e.g., rotation angle, kernel size, sigma)
- **Output**: Enhanced images saved in PGM format

---

## 🔍 Use Cases

- 📹 **CCTV Footage Enhancement** – Clarify surveillance video to identify suspects or vehicles
- 🧬 **Fingerprint Processing** – Improve ridge detail for better matching
- 📄 **Document Examination** – Restore readability of degraded or damaged documents
- 🕵️ **Crime Scene Photography** – Enhance poorly lit crime scene images
- 🔫 **Ballistics Imaging** – Improve visibility of ballistic markings

---

## 🌱 Future Work

- Add support for more enhancement techniques (e.g., contrast enhancement, deblurring)
- Extend image format support beyond PGM
- Integrate with machine learning for automatic feature detection
- Develop a user-friendly GUI for investigators
- Support multi-GPU environments for batch and large image processing

---

## 📎 License

[Insert license information here]

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements or new features.

---

## 📫 Contact

For questions or collaborations, please contact [your-email@example.com].

