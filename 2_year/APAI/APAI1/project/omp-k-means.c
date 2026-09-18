/**********************************************************************************************
 * OpenMP implementation of K-Means
 * 
 * Compile with:
 *      1) Baseline
 *          gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *      2) Baseline + optimization
 *          gcc -std=c99 -Wall -Wpedantic -fopenmp -O3 omp-k-means.c -o omp-k-means
 *      3) Baseline + optimization + SIMD
 *          gcc -std=c99 -Wall -Wpedantic -fopenmp -O3 -DUSE_SIMD omp-k-means.c -o omp-k-means
 *
 * Run with:
 *      ./omp-k-means K input_file output_file
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

/* Necessary to turn on/off the use of SIMD pragmas using a single file */
#ifdef USE_SIMD
#define SIMD_PRAGMA      _Pragma("omp simd")
#define SIMD_REDUCTION   _Pragma("omp simd reduction(+:result)")
#else
#define SIMD_PRAGMA
#define SIMD_REDUCTION
#endif

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>     // for memset()

int n_dims;             /* number of dimensions.                */ 

int n_points;           /* number of data points.               */

int n_clusters;         /* number of clusters.                  */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid, which is an
                           array of `n_dims` floating point
                           numbers.                             */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points that belong to
                           cluster j.                           */

int *cluster_of;        /* [array of length n_points] `clusters_of[i]`
                           is the ID of the cluster assigned to the
                           i-th data point; cluster IDs are integer in
                           0..(n_clusters-1).                   */


/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/*
    NOTE:

    -   SIMD vectorization targets the innermost loop over n_dims, which has
        stride-1 memory access and no cross-iteration dependencies.

    -   The pragma itself doesn't produce the vectorization. It is a hint for the
        compiler to try to vectorize the loops that otherwise would not be vectorized. 
        This could be checked with the specific report using `-fopt-info-vec-optimized`:

            -   without optimization flags, no loops are vectorized at all. The pragma
                is ignored without a sufficient optimization level.

            -   with `-O3` only some functions triggered the "loop versioned for vectorization 
                because of possible aliasing". The compiler cannot prove that float *p1 and *p2 
                do not overlap, so it generates both a vectorized and a scalar version of the 
                loop, selecting at runtime, adding overhead on every call. 
                
                NOTE: we still added the pragma to all the operations for coherence.

            -   with `-O3 -DUSE_SIMD` the aliasing warnings disappear entirely, so the compiler 
                uses a single clean vectorized path.
*/

/* Set all components of vector `p` of size `n_dims` equal to zero. 
   
    NOTE: this will not be used since it does the same thing ot memset(), 
    even if it does not account for a meaningful time reduction.
*/
void vzero( float * p )
{
    SIMD_PRAGMA
    for (int d = 0; d < n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd( float * p1, const float * p2 )
{
    SIMD_PRAGMA
    for (int d = 0; d < n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float * p, float v )
{
    SIMD_PRAGMA
    for (int d = 0; d < n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float * p1, const float * p2 )
{
    SIMD_PRAGMA
    for (int d = 0; d < n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist( float *p1, float *p2 )
{
    float result = 0.0;
    SIMD_REDUCTION
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
int IDX(int i, int d)
{
    return i*n_dims + d;
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/* Assign each data point to the nearest centroid. Updates
   the `counts` array. */
void classify( void )
{

    /*  NOTES:

        -   Given the variable ranges so that we can say n_points >> n_dims and 
            n_points >> n_clusters. For this reason we parallelize only the outer 
            loop on n_points, which is the most expensive. Moreover it also has 
            no cross-iteration dependencies, so it is ideal for parallelization.

        -   In general loops over n_clusters are both small and with dependancies so we 
            didn't consider them for optimization.

        -   Since we have OpenMP 4.5 on the university cluster we use array reduction 
            on counts to avoid race conditions without atomic/critical overhead.
        
        -   We leave schedule(static) since each point always does n_cluster comparisons 
            over n_dim dimensions, so the workload is constant.
        
    */

    /* Same of looping and setting to 0*/
    memset(counts, 0, n_clusters * sizeof(*counts));

    /* Parallelize over N */
    #pragma omp parallel for default(none) \
        shared(data, centroids, cluster_of, n_points, n_clusters, n_dims) \
        reduction(+:counts[:n_clusters]) \
        schedule(static)
    for (int i=0; i<n_points; i++) {
        /* Index and squared distance of the nearest centroid. */
        int nearest = 0;
        float mindist = sqdist( &data[IDX(i,0)], &centroids[IDX(0,0)] );
        for (int j=1; j<n_clusters; j++) {
            const float dist = sqdist( &data[IDX(i, 0)], &centroids[IDX(j, 0)] );
            if ( dist < mindist ) {
                mindist = dist;
                nearest = j;
            }
        }
        /* Assign the point to the nearest centroid and update cluster size. */
        cluster_of[i] = nearest;
        counts[nearest]++;
    }
}

/* Update the centroids. Set the centroid of each cluster to the
   barycenter of the points. Returns the maximum shift, i.e., the
   maximum difference between the (squared) old and new position of
   all centroids. */
float update_centroids( void )
{

    /*  NOTES:

        -   We apply the same reasoning of classify(), so we parallelize only the loop over 
            n_points, which is the most expensive one, and we use array reduction to update 
            new_centroids without race conditions.

        -   We could parallelize over n_clusters but it would require max reduction for a 
            very small number of iterations, so we don't consider it worth it.

        -   It is interesting to notice that we tried to move this loop inside of classify(),
            since we are performing the same parallelization and we could save some time by 
            doing only one parallel region and one loop over N, but it surprisingly did not
            resulted in an improvement, so we left it separated for code clarity. On the other
            hand we noticed that the serial part of this function takes 0.000s quite constantly.
    */

    /* Same of looping `vzero` */
    memset(new_centroids, 0, n_clusters * n_dims * sizeof(*new_centroids));

    /* Parallelize over N */
    #pragma omp parallel for default(none) \
        shared(data, cluster_of, n_points, n_clusters, n_dims) \
        reduction(+: new_centroids[:n_clusters * n_dims]) \
        schedule(static)
    for (int i = 0; i < n_points; i++) {
        vadd( &new_centroids[IDX(cluster_of[i], 0)], &data[IDX(i, 0)] );
    }

    float maxshift = 0.0f;
    for (int j=0; j<n_clusters; j++) {
        /* If a cluster is empty, we simply copy the old centroid to
           the new one. */
        if (counts[j] == 0) {
            vcopy( &new_centroids[IDX(j,0)], &centroids[IDX(j,0)] );
        } else {
            vmul( &new_centroids[IDX(j, 0)], 1.0f/counts[j] );
        }
        const float shift = sqdist( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
        if (shift > maxshift)
            maxshift = shift;
        vcopy( &centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)] );
    }

    return maxshift;
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
void read_input( FILE *f )
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

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids into a
   file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time. */
void save_centroids( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j=0; j<n_clusters; j++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters
   into a file.

   This function is useful for generating a movie showing how the
   centroids get updated, otherwise it can be omitted.

   This function can be enable by defining the MAKE_MOVIE symbol at
   compilation time.
*/
void save_clusters( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "out_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    // const float TOL = 1e-5;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

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

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    float shift;
    int iter = 0;
    const double tstart = hpc_gettime();
    double t_classify = 0.0;
    double t_update = 0.0;

    do {
        double t0 = hpc_gettime();
        classify();
        t_classify += hpc_gettime() - t0;

        t0 = hpc_gettime();
        shift = update_centroids();
        t_update += hpc_gettime() - t0;

        iter++;
    /*  NOTE: 

        As said in the instructions we were given, we have to remove 
        the tolerance so that the workload O(N*D*K*I) has I fixed.
    */
    // } while ((shift > TOL) && (iter <= MAXITER));
    } while (iter < MAXITER);

    const double elapsed = hpc_gettime() - tstart;

    // They will be extracted by the bash launcher and collected into a .csv
    printf("tot: %.3f\n", elapsed);
    printf("t_classify: %.3f\n", t_classify);
    printf("t_update: %.3f\n", t_update);
    printf("iter: %d\n", iter);
    printf("final_shift: %f\n", shift);

    save_results(outputf);

    fclose(outputf);

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}