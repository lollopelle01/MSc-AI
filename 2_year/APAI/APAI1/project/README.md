### 1. OpenMP Version
The OpenMP version supports standard parallelization, compiler optimizations, and SIMD vectorization.

**Compilation:**
* **Baseline:**
    `gcc -std=c99 -Wall -fopenmp omp-k-means.c -o omp-k-means`
* **With O3 Optimizations:**
    `gcc -std=c99 -Wall -fopenmp -O3 omp-k-means.c -o omp-k-means`
* **With O3 and SIMD:**
    `gcc -std=c99 -Wall -fopenmp -O3 -DUSE_SIMD omp-k-means.c -o omp-k-means`

**Usage:**
```bash
./omp-k-means <K> <input_file> <output_file>
```

---

### 2. CUDA Version
The CUDA version supports both **AoS** (Array of Structures) and **SoA** (Structure of Arrays) memory layouts.

**Compilation:**
* **AoS Layout (Default):**
    `nvcc cuda-k-means.cu -o cuda-k-means`
* **SoA Layout:**
    `nvcc -DUSE_SOA cuda-k-means.cu -o cuda-k-means`

**Usage:**
```bash
./cuda-k-means <K> <input_file> <output_file> <reduction_type>
```
* `K`: Number of clusters.
* `reduction_type`: 
    * `0`: Use **Atomic** reduction.
    * `1`: Use **Binary-halving tree** reduction.
