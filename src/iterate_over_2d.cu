__global__ void MyKernel(float *dev_ptr, size_t pitch, int width, int height) {
  for (int r = 0; r < height; ++r) {
    float *row = (float *)((char *)dev_ptr + r * pitch);
    for (int c = 0; c < width; ++c) {
      float element = row[c];
    }
  }
}

int main() {

  int width = 64, height = 64;
  float *dev_ptr;
  size_t pitch;

  cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(float), height);
  MyKernel<<<100, 512>>>(dev_ptr, pitch, width, height);
}
