/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
       Device kernels of the derivatives3d benchmark, ROCm build
*/

#include <complex>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

namespace{
void check(hipError_t status, char const *what){
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

__global__ void multiply_kernel(long n, double2 const *factor, double2 const *in, double2 *out){
    long const i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n){
        double2 const a = factor[i];
        double2 const b = in[i];
        out[i] = make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }
}
}

//! \brief out = factor * in, entrywise, on the default stream.
void device_multiply(long n, std::complex<double> const factor[], std::complex<double> const in[], std::complex<double> out[]){
    if (n == 0) return;
    int const threads = 256;
    hipLaunchKernelGGL(multiply_kernel, dim3((n + threads - 1) / threads), dim3(threads), 0, 0, n,
                       reinterpret_cast<double2 const*>(factor), reinterpret_cast<double2 const*>(in),
                       reinterpret_cast<double2*>(out));
    check(hipGetLastError(), "multiply_kernel");
}

//! \brief Copies between any two of host and device memory.
void device_copy(void *destination, void const *source, size_t bytes){
    if (bytes > 0) check(hipMemcpy(destination, source, bytes, hipMemcpyDefault), "hipMemcpy");
}

void device_sync(){
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}
