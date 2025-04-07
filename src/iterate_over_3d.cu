__global__ void MyKernel(cudaPitchedPtr dev_pitcher_ptr, int width, int height,
                         int depth) {
  char *dev_ptr = (char *)dev_pitcher_ptr.ptr; // Why cast in char*?
  size_t pitch = dev_pitcher_ptr.pitch; // This is in BYTES (including padding)
  size_t slice_pitch =
      pitch *
      height; // Height is in number of elements -> slice_pitch is in bytes

  for (int z = 0; z < depth; ++z) {
    char *slice =
        dev_ptr + z * slice_pitch; // This moves the pointer by z * slice_pitch
                                   // bytes, which is z * slice_pitch * 1
    for (int y = 0; y < height; ++y) {
      float *row = (float *)(slice + y * pitch);
      for (int x = 0; x < width; ++x) {
        float element = row[x];
      }
    }
  }
}

int main() {
  int width = 64, height = 64, depth = 64;
  // cudaExtent describes the dimensions of a 3D array or a memory region
  cudaExtent extent =
      make_cudaExtent(width * sizeof(float), height,
                      depth); // w: in bytes when referring to linear memory

  // cudaPitchedPtr is a struct that describes a pitched memory region ( pitched
  // -> padded 2D or 3D array for alignment)
  cudaPitchedPtr dev_pitcher_ptr;
  cudaMalloc3D(&dev_pitcher_ptr, extent);
  MyKernel<<<100, 512>>>(dev_pitcher_ptr, width, height, depth);
}
