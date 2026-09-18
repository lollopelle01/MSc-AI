/**********************************************************************************************
 * CUDA implementation of K-Means
 * 
 * Compile with:
 *      1) AoS 
 *          nvcc cuda-k-means.cu -o cuda-k-means
 *      2) SoA
 *          nvcc -DUSE_SOA cuda-k-means.cu -o cuda-k-means
 *
 * Run with:
 *      1) Atomics reduction
 *          ./cuda-k-means K input_file output_file 0
 *      2) Binary-halving tree reduction
 *          ./cuda-k-means K input_file output_file 1
 *
 **********************************************************************************************/

/* Enable POSIX extensions which are required for making function
   `clock_gettime()` (used in "hpc.h") visible. The following
   `#define` must come at the very beginning, before including
   anything else.
*/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // for memcpy()

/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *p = malloc(size);
    assert(p != NULL);
    return p;
}

/******************************************************************************
 **
 ** GLOBAL VARIABLES
 **
 ******************************************************************************/

int n_dims, n_points, n_clusters;

/* Runtime kernel selector: 0 = atomics, 1 = tree reduction */
int use_reduction;

/*  Per-kernel block sizes, selected via occupancy API in select_block_dims().
    Each of the four kernels has a different shared-memory footprint and
    therefore its own optimal block size.                                       */
int block_dim_classify, block_dim_update;

/*  Number of dimensions processed per tile in update_centroids_kernel_reduction.
    We use it so that we can permit larger block sizes, which are beneficial for
    occupancy, even when D is large and would not fit in shared memory otherwise.
    It is bounded to the shared memory limit.                                     */
int tile_dims_update;

/* Host arrays
   Always AoS at allocation time. data[] is transposed to
   SoA in-place before the GPU phase (if USE_SOA) and restored
   to AoS after.                                                                  */
float *data, *centroids, *new_centroids;
int *counts, *cluster_of;

/* Device arrays
   They mirror the host arrays on the GPU.                                        */
float *d_data, *d_centroids, *d_new_centroids;
int *d_counts, *d_cluster_of;

/******************************************************************************
 **
 ** INDEX HELPERS
 **
 ******************************************************************************/

/*  NOTE:

    -   They are both callable from the host and from the device, this
        refactoring is applied to all the functions in the file.

    -   With these functions we can easily switch between AoS and SoA layout
        for data[]:
        -   AoS: data is NxD matrix stored in row-major order (default)
        -   SoA: data is DxN matrix stored in row-major order (defined by USE_SOA)

        Since the inner loops are over D, the SoA layout allows coalesced memory
        access on the GPU. For doing this data[] is transposed in-place on the host
        before the GPU phase (if USE_SOA).

    -   centroids[] is always AoS on the host and on the device. Each thread in a
        block would read the same centroid at each iteration, so it wouldn't benefit
        from SoA.
*/

// ONLY FOR data[]
__host__ __device__ int IDX_data(int i, int d, int nd, int np)
{
#ifdef USE_SOA
    return d * np + i;
#else
    return i * nd + d;
#endif
}

// FOR ALL NON-SoA STRUCTURES
__host__ __device__ int IDX_aos(int j, int d, int nd)
{
    return j * nd + d;
}


/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero(float *p)
{
    for (int d = 0; d < n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd(float *p1, const float *p2)
{
    for (int d = 0; d < n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul(float *p, float v)
{
    for (int d = 0; d < n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy(float *p1, const float *p2)
{
    for (int d = 0; d < n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist(float *p1, float *p2)
{
    float result = 0.0;
    for (int d = 0; d < n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/* It is like `sqdist` but for device code */
__device__ float sqdist_d(const float *data, const float *centroids,
                           int i, int j, int nd, int np)
{
    float result = 0.0f;
    for (int d = 0; d < nd; d++) {
        float diff = data[IDX_data(i, d, nd, np)]
                   - centroids[IDX_aos(j, d, nd)];
        result += diff * diff;
    }
    return result;
}

/* Layout transposition functions for enabling SoA */
#ifdef USE_SOA
static void transpose_aos_to_soa(void)
{
    float *tmp = (float *)safe_malloc(n_points * n_dims * sizeof(float));
    for (int i = 0; i < n_points; i++)
        for (int d = 0; d < n_dims; d++)
            /* AoS source -> SoA destination */
            tmp[IDX_data(i, d, n_dims, n_points)] = data[IDX_aos(i, d, n_dims)];
    memcpy(data, tmp, n_points * n_dims * sizeof(float));
    free(tmp);
}

static void transpose_soa_to_aos(void)
{
    float *tmp = (float *)safe_malloc(n_points * n_dims * sizeof(float));
    for (int d = 0; d < n_dims; d++)
        for (int i = 0; i < n_points; i++)
            /* SoA source -> AoS destination */
            tmp[IDX_aos(i, d, n_dims)] = data[IDX_data(i, d, n_dims, n_points)];
    memcpy(data, tmp, n_points * n_dims * sizeof(float));
    free(tmp);
}
#endif

/******************************************************************************
 **
 ** CUDA KERNELS - first version (atomics on shared memory)
 **
 ******************************************************************************/

/*  NOTE:

    -   The two kernels in this section use shared-memory atomics to
        accumulate into block-local copies of the contended structures,
        then flush to global memory with one atomicAdd per slot.

    -   The latency of shared-memory atomics is much lower than that of
        global atomics, so this approach is faster than using only global
        atomics directly. However, it requires more shared memory per block,
        which can reduce SM occupancy at large K and D.

    -   The shared memory will be allocated dynamically when calling the kernel,
        since it depends on D and K. The block dimension and the number of blocks
        will be computed to maximize the occupancy, using a CUDA API. 
        
    -   We do not guard against shared memory overflow since since it did not occur 
        in our ranges and in general we assumed K and D small.
*/

__global__ void classify_kernel_atomic(
        const float *data,
        const float *centroids,
        int *cluster_of,
        int *counts,
        int n_points, int n_clusters, int n_dims)
{
    /* Block-local count array. K threads parallelize its initialization. */
    extern __shared__ int s_counts[];
    for (int j = threadIdx.x; j < n_clusters; j += blockDim.x)
        s_counts[j] = 0;

    /* all slots must be 0 before any write */
    __syncthreads();

    /* Each thread takes a data point and compares it to all centroids. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_points) {
        int nearest = 0;
        float mindist = sqdist_d(data, centroids, i, 0, n_dims, n_points);
        for (int j = 1; j < n_clusters; j++) {
            float dist = sqdist_d(data, centroids, i, j, n_dims, n_points);
            if (dist < mindist) {
                mindist = dist;
                nearest = j;
            }
        }
        cluster_of[i] = nearest;
        atomicAdd(&s_counts[nearest], 1); /* shared-memory atomic */
    }

    /* all threads must have written before flush */
    __syncthreads();

    /* One global atomicAdd per (block, cluster). */
    for (int j = threadIdx.x; j < n_clusters; j += blockDim.x)
        atomicAdd(&counts[j], s_counts[j]);
}

__global__ void update_centroids_kernel_atomic(
        const float *data,
        const int *cluster_of,
        float *new_centroids,
        int n_points, int n_clusters, int n_dims)
{
    /* Block-local accumulator: K*D floats. K*D threads parallelize its initialization. */
    extern __shared__ float s_new_centroids[];
    for (int j = threadIdx.x; j < n_clusters * n_dims; j += blockDim.x)
        s_new_centroids[j] = 0.0f;

    /* all slots must be 0 before any write */
    __syncthreads();

    /* Each thread takes a data point and accumulates its D dimensions. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_points) {
        const int c = cluster_of[i];
        for (int d = 0; d < n_dims; d++) {
            atomicAdd(
                &s_new_centroids[IDX_aos(c, d, n_dims)],
                data[IDX_data(i, d, n_dims, n_points)]
            );
        }
    }

    /* all threads must have written before flush */
    __syncthreads();

    /* One global atomicAdd per (block, cluster, dimension). */
    for (int j = threadIdx.x; j < n_clusters * n_dims; j += blockDim.x)
        atomicAdd(&new_centroids[j], s_new_centroids[j]);
}

/******************************************************************************
 **
 ** CUDA KERNELS - second version (binary-halving tree reduction)
 **
 ******************************************************************************/

/*  NOTE:

    -   A crucial requirement for this reduction is that the block dimension has
        to be a power of two. Since we choose the optimal block dimension via the
        occupancy API, we round it to the nearest power of two if it is not already
        a power of two. Of course this could lead to inefficiencies.

    -   In update_centroids_reduction, by simply applying the tree reduction we would 
        pay too much in synchronization, so the shared memory was increased to compute 
        multiple slots in parallel and reduce them in a single pass, minimizing the 
        number of __syncthreads() calls. The shared memory size then became a problem 
        so we introduce tiling not to exceed the limit and still maintaining the same 
        strategy. 
        We covered the shared memory overflow here since the block size factor that 
        was added to the equation but, for the same reason as the atomic kernels, we 
        did not covered it in classify_kernel_reduction.

    -   Here we do not use atomics on shared memory at all, so we don't have their
        contention even if it was already small. The global atomics are the same. We
        are paying a higher price only in terms of synchronization, which is however
        now minimized by the parallel reduction strategy described above.

*/

__global__ void classify_kernel_reduction(
        const float *data,
        const float *centroids,
        int *cluster_of,
        int *counts,
        int n_points, int n_clusters, int n_dims)
{
    extern __shared__ int s_reduce_int[];

    const int li = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + li;

    /* Each thread finds the nearest centroid of its data point */
    int nearest = -1; /* out-of-range threads do not count */
    if (i < n_points) {
        nearest = 0;
        float mindist = sqdist_d(data, centroids, i, 0, n_dims, n_points);
        for (int j = 1; j < n_clusters; j++) {
            float dist = sqdist_d(data, centroids, i, j, n_dims, n_points);
            if (dist < mindist) {
                mindist = dist;
                nearest = j;
            }
        }
        cluster_of[i] = nearest;
    }

    /* Fill all K buffers in parallel: one write per (thread, cluster) */
    for (int c = 0; c < n_clusters; c++)
        s_reduce_int[c * blockDim.x + li] = (nearest == c) ? 1 : 0;

    /* all K*BLOCK slots must be written before any read */
    __syncthreads();

    /* Single reduction pass over all K buffers simultaneously */
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (li < stride)
            for (int c = 0; c < n_clusters; c++)
                s_reduce_int[c * blockDim.x + li] += s_reduce_int[c * blockDim.x + li + stride];
        __syncthreads(); // all threads must wait
    }

    /* One global atomicAdd per (block, cluster) */
    if (li == 0)
        for (int c = 0; c < n_clusters; c++)
            atomicAdd(&counts[c], s_reduce_int[c * blockDim.x]);
}

__global__ void update_centroids_kernel_reduction(
        const float *data,
        const int *cluster_of,
        float *new_centroids,
        int n_points, int n_clusters, int n_dims,
        int tile_dims)
{
    extern __shared__ float s_reduce_float[];

    const int li = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + li;

    /* Cache cluster assignment once, avoiding multiple re-reads */
    const int my_cluster = (i < n_points) ? cluster_of[i] : -1;

    for (int c = 0; c < n_clusters; c++) {

        /* Iterate over tiles of dimensions */
        for (int d0 = 0; d0 < n_dims; d0 += tile_dims) {

            /* Dimensions in this tile: the last tile may be smaller so we give the remaining */
            const int tile = (d0 + tile_dims < n_dims) ? tile_dims : n_dims - d0;

            /* Fill tile parallel buffers, one write per (thread, local-dim) */
            for (int t = 0; t < tile; t++) {
                float val = 0.0f;
                if (my_cluster == c)
                    val = data[IDX_data(i, d0 + t, n_dims, n_points)];
                s_reduce_float[t * blockDim.x + li] = val;
            }

            /* all tile*BLOCK slots must be written before any read */
            __syncthreads();

            /* Single reduction pass over all tile buffers simultaneously */
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                if (li < stride)
                    for (int t = 0; t < tile; t++)
                        s_reduce_float[t * blockDim.x + li] += s_reduce_float[t * blockDim.x + li + stride];
                __syncthreads(); // all threads must wait
            }

            /* One global atomicAdd per (block, cluster, dimension) */
            if (li == 0)
                for (int t = 0; t < tile; t++)
                    atomicAdd(
                        &new_centroids[IDX_aos(c, d0 + t, n_dims)],
                        s_reduce_float[t * blockDim.x]
                    );
        }
    }
}

/******************************************************************************
 **
 ** Host wrappers that launch kernels.
 **
 ******************************************************************************/

void classify(void)
{
    const int BLOCK = block_dim_classify;
    const int GRID = (n_points + BLOCK - 1) / BLOCK;
    int SMEM;

    /* The same of looping and setting to 0 */
    cudaSafeCall(cudaMemset(d_counts, 0, n_clusters * sizeof(int)));

    /* Copy centroids to device */
    cudaSafeCall(cudaMemcpy(d_centroids, centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyHostToDevice));

    if (use_reduction) {
        SMEM = n_clusters * BLOCK * sizeof(int);
        classify_kernel_reduction<<<GRID, BLOCK, SMEM>>>(
            d_data, d_centroids, d_cluster_of, d_counts,
            n_points, n_clusters, n_dims
        );
    } else {
        SMEM = n_clusters * sizeof(int);
        classify_kernel_atomic<<<GRID, BLOCK, SMEM>>>(
            d_data, d_centroids, d_cluster_of, d_counts,
            n_points, n_clusters, n_dims
        );
    }
    cudaCheckError();

    /* Copy the final cluster counts to the host */
    cudaSafeCall(cudaMemcpy(counts, d_counts, n_clusters * sizeof(int), cudaMemcpyDeviceToHost));
}

float update_centroids(void)
{
    const int BLOCK = block_dim_update;
    const int GRID = (n_points + BLOCK - 1) / BLOCK;
    int SMEM;

    /* It does the same thing as `vzero`, but it is optimized */
    cudaSafeCall(cudaMemset(d_new_centroids, 0, n_clusters * n_dims * sizeof(float)));

    if (use_reduction) {
        SMEM = tile_dims_update * BLOCK * sizeof(float);
        update_centroids_kernel_reduction<<<GRID, BLOCK, SMEM>>>(
            d_data, d_cluster_of, d_new_centroids,
            n_points, n_clusters, n_dims,
            tile_dims_update
        );
    } else {
        SMEM = n_clusters * n_dims * sizeof(float);
        update_centroids_kernel_atomic<<<GRID, BLOCK, SMEM>>>(
            d_data, d_cluster_of, d_new_centroids,
            n_points, n_clusters, n_dims
        );
    }
    cudaCheckError();

    /* Copy the new centroids to the host */
    cudaSafeCall(cudaMemcpy(new_centroids, d_new_centroids, n_clusters * n_dims * sizeof(float), cudaMemcpyDeviceToHost));

    /*  Normalise and compute max centroid shift (host-side, always AoS). 
        
        NOTE: when we tried to merge the previous loop in the classify one, we 
        noticed that the following loop takes 0.000s, any change would not give
        any improvement.
    */
    
    float maxshift = 0.0f;
    for (int j = 0; j < n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy(
                &new_centroids[IDX_aos(j, 0, n_dims)],
                &centroids[IDX_aos(j, 0, n_dims)]
            );
        } else {
            vmul(
                &new_centroids[IDX_aos(j, 0, n_dims)],
                1.0f / counts[j]
            );
        }

        float shift = sqdist(
            &centroids[IDX_aos(j, 0, n_dims)],
            &new_centroids[IDX_aos(j, 0, n_dims)]
        );
        if (shift > maxshift)
            maxshift = shift;

        vcopy(
            &centroids[IDX_aos(j, 0, n_dims)],
            &new_centroids[IDX_aos(j, 0, n_dims)]
        );
    }
    return maxshift;
}

/******************************************************************************
 **
 ** Block size selection.
 **
 ******************************************************************************/

/*
    A utility function to guarantee the block size to be a power of 2.
    We choose the nearest power of 2 since we want the nearest approximation
    of the ideal block dimension returned by the occupancy API. Choosing the
    next power of 2 could lead to a much higher block dimension and therefore
    to a much lower occupancy.

    NOTE: we chose the lower power of 2 in case of tie, since the CUDA API tends
    to give the highest number (so the minimum number of blocks) that maximizes 
    occupancy (in order to minimize scheduling overhead than), so a bigger block 
    size would give not the maximum occupancy but a suboptimal one. By doing this 
    we increase the possibility to still have the maximum occupancy but maybe with 
    a smaller block size, which is not that much a problem.
    Anyway the kernels usually require a dimension of 768 so the decision is
    between 512 and 1024 (first one better than second one).
*/
static int nearest_pow2(int x)
{
    int lo = 1;
    while (lo * 2 < x)
        lo *= 2;
    int hi = lo * 2;
    return (x - lo <= hi - x) ? lo : hi;
}

/*  NOTE:
    -   The reduction kernels require a power-of-two block size, so their
        block dim is rounded to the nearest power of two after the API call.

    -   The smem of the reduction kernels depends on the block size itself,
        creating a circular dependency with the occupancy API. We break it
        by passing a lower-bound estimate (minimum warp size * slots) as seed.
        This may cause the API to suggest a block size that is not optimal.
*/
static void select_block_dims(size_t smem_limit,
                              int *out_min_grid_classify,
                              int *out_min_grid_update)
{
    size_t smem_classify, smem_update;

    if (use_reduction) {

        /* classify kernel */
        smem_classify = (size_t) n_clusters * 32 * sizeof(int);
        cudaSafeCall(cudaOccupancyMaxPotentialBlockSize(
            out_min_grid_classify, &block_dim_classify,
            classify_kernel_reduction,
            smem_classify, 0
        ));

        /* update kernel */
        smem_update = (size_t) n_dims * 32 * sizeof(float);
        cudaSafeCall(cudaOccupancyMaxPotentialBlockSize(
            out_min_grid_update, &block_dim_update,
            update_centroids_kernel_reduction,
            smem_update, 0
        ));

        /* Reduction kernels require a power-of-two block size */
        block_dim_classify = nearest_pow2(block_dim_classify);
        block_dim_update = nearest_pow2(block_dim_update);
        assert(block_dim_classify <= 1024);
        assert(block_dim_update <= 1024);

        /* Compute the tile dimensions (capped to D) */
        tile_dims_update = (int)(smem_limit / ((size_t)block_dim_update * sizeof(float)));
        if (tile_dims_update > n_dims)
            tile_dims_update = n_dims;
        assert(tile_dims_update >= 1);

    } else {

        /* classify kernel */
        smem_classify = (size_t) n_clusters * sizeof(int);
        cudaSafeCall(cudaOccupancyMaxPotentialBlockSize(
            out_min_grid_classify, &block_dim_classify,
            classify_kernel_atomic,
            smem_classify, 0
        ));

        /* update kernel */
        smem_update = (size_t) n_clusters * n_dims * sizeof(float);
        cudaSafeCall(cudaOccupancyMaxPotentialBlockSize(
            out_min_grid_update, &block_dim_update,
            update_centroids_kernel_atomic,
            smem_update, 0
        ));
    }
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input(FILE *f)
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_points % n_dims == 0); /* If this assertion fails, then
                                       there is some line of the input
                                       file that has != n_dims
                                       items. */

    n_points = n_items / n_dims;

    data = (float *)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX_aos(i, d, n_dims)]);
            assert(nread == 1);
        }
    }
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b)
{
    return a + rand() % (b - a + 1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids(void)
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy(
                &centroids[IDX_aos(select, 0, n_dims)],
                &data[IDX_aos(i, 0, n_dims)]
            );
        }
        remaining--;
    }
}

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results(FILE *f)
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j = 0; j < n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d = 0; d < n_dims; d++)
            fprintf(f, " %f", centroids[IDX_aos(j, d, n_dims)]);
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < n_dims; d++)
            fprintf(f, "%f ", data[IDX_aos(i, d, n_dims)]);
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

int main(int argc, char *argv[])
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s K input output [use_reduction]\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);
    use_reduction = (argc == 5) ? atoi(argv[4]) : 0;

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n", n_dims);
    fprintf(outputf, "# Clusters: %d\n", n_clusters);

    /* Host allocation */
    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

#ifdef USE_SOA
    /* Transpose data[] from AoS to SoA once, before the GPU phase */
    transpose_aos_to_soa();
#endif

    /*  Query the device for the actual shared memory limit per block to have a more robust approach */
    int smem_per_block_int;
    cudaSafeCall(cudaDeviceGetAttribute(&smem_per_block_int, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    const size_t smem_limit = (size_t)smem_per_block_int;

    /*  Select block sizes for the active kernel pair and compute tile_dims_update.
        The minimum grid sizes for the classify and update kernels are reported for
        the .csv.                                                                    */
    int min_grid_classify, min_grid_update;
    select_block_dims(smem_limit, &min_grid_classify, &min_grid_update);

    /* Effective grid sizes, used for the .csv report */
    const int grid_classify = (n_points + block_dim_classify - 1) / block_dim_classify;
    const int grid_update = (n_points + block_dim_update - 1) / block_dim_update;

    /* Device allocation */
    cudaSafeCall(cudaMalloc(&d_data, n_points * n_dims * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_centroids, n_clusters * n_dims * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_new_centroids, n_clusters * n_dims * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_cluster_of, n_points * sizeof(int)));
    cudaSafeCall(cudaMalloc(&d_counts, n_clusters * sizeof(int)));

    /* Copy data[] to device */
    cudaSafeCall(cudaMemcpy(d_data, data, n_points * n_dims * sizeof(float), cudaMemcpyHostToDevice));

    /* K-means loop */
    float shift;
    int iter = 0;
    double t_classify = 0.0;
    double t_update = 0.0;
    const double tstart = hpc_gettime();
    do {
        double t0 = hpc_gettime();
        classify();
        t_classify += hpc_gettime() - t0;

        t0 = hpc_gettime();
        shift = update_centroids();
        t_update += hpc_gettime() - t0;

        iter++;
    } while (iter < MAXITER);

    const double elapsed = hpc_gettime() - tstart;

    /* Print results for the .csv */
    printf("tot: %.6f\n", elapsed);
    printf("t_classify: %.6f\n", t_classify);
    printf("t_update: %.6f\n", t_update);
    printf("iter: %d\n", iter);
    printf("final_shift: %f\n", shift);
    printf("block_dim_classify: %d\n", block_dim_classify);
    printf("block_dim_update: %d\n", block_dim_update);
    printf("grid_classify: %d\n", grid_classify);
    printf("grid_classify_optimal: %d\n", min_grid_classify);
    printf("grid_update: %d\n", grid_update);
    printf("grid_update_optimal: %d\n", min_grid_update);

    /* Copy back the result */
    cudaSafeCall(cudaMemcpy(cluster_of, d_cluster_of, n_points * sizeof(int), cudaMemcpyDeviceToHost));

#ifdef USE_SOA
    /* Restore AoS layout so save_results() can read via IDX_aos. */
    transpose_soa_to_aos();
#endif

    save_results(outputf);

    fclose(outputf);

    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaFree(d_centroids));
    cudaSafeCall(cudaFree(d_new_centroids));
    cudaSafeCall(cudaFree(d_cluster_of));
    cudaSafeCall(cudaFree(d_counts));

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}
