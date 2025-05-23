// nv12_to_rgb_normalized.cu
#include <cstdint>

__device__ void yuv_to_normalized_rgb_bt709(
    uint8_t y, uint8_t u, uint8_t v, 
    float& r, float& g, float& b
) {
    // Convert YUV to float RGB (BT.709)
    float fy = (static_cast<float>(y) - 16.0f) / 219.0f;
    float fu = (static_cast<float>(u) - 128.0f) / 224.0f;
    float fv = (static_cast<float>(v) - 128.0f) / 224.0f;

    // BT.709 coefficients
    r = 1.164f * fy + 1.793f * fv;
    g = 1.164f * fy - 0.534f * fv - 0.213f * fu;
    b = 1.164f * fy + 2.115f * fu;

    // Clamp to [0.0, 1.0]
    r = fmaxf(0.0f, fminf(r, 1.0f));
    g = fmaxf(0.0f, fminf(g, 1.0f));
    b = fmaxf(0.0f, fminf(b, 1.0f));
}

extern "C" __global__ void nv12_to_normalized_rgb_kernel(
    const uint8_t* nv12,
    float* rgb,          // Output as float32 RGB (3 channels per pixel)
    int width,
    int height,
    int rgb_pitch        // Pitch in bytes (width * 3 * sizeof(float))
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + (width * height);

    // Read YUV values
    uint8_t y_val = y_plane[y * width + x];
    int uv_x = x / 2;
    int uv_y = y / 2;
    uint8_t u_val = uv_plane[uv_y * width + 2 * uv_x];
    uint8_t v_val = uv_plane[uv_y * width + 2 * uv_x + 1];

    // Convert to normalized RGB
    float r, g, b;
    yuv_to_normalized_rgb_bt709(y_val, u_val, v_val, r, g, b);

    // Write output (interleaved RGB, float32)
    float* rgb_pixel = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(rgb) + y * rgb_pitch + x * 3 * sizeof(float)
    );
    rgb_pixel[0] = r;
    rgb_pixel[1] = g;
    rgb_pixel[2] = b;
}

// Add this to your .cu file
extern "C" __global__ void interleaved_to_chw_kernel(
    const float* rgb,
    float* chw_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int rgb_idx = (y * width + x) * 3;
    int spatial_idx = y * width + x;
    
    chw_output[spatial_idx] = rgb[rgb_idx];               // R channel
    chw_output[width*height + spatial_idx] = rgb[rgb_idx+1]; // G channel
    chw_output[2*width*height + spatial_idx] = rgb[rgb_idx+2]; // B channel
}
