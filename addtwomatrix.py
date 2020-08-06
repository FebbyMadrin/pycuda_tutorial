import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

m=5
n=5

a = numpy.random.rand(m,n)
a = a.astype(numpy.float32)

b = numpy.random.rand(m,n)
b = b.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
print (a)
print (b)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

module = SourceModule("""
    __global__ void add(float *a, float *b)
    {
     int idx = threadIdx.x + threadIdx.y*blockDim.x;
     a[idx]=a[idx] + b[idx];
    }                      
    """)

fx = module.get_function("add")
fx(a_gpu, b_gpu, block=(m,n,1))

Add = numpy.empty_like(a)
cuda.memcpy_dtoh(Add, a_gpu)

print (Add)