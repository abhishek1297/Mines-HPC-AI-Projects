/*
* Author: Abhishek Purandare
* Course: UE3 High-Performance Computing
*
* The file contains exercises 1 to 4
*/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <immintrin.h>

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)
#define T 10
#define N ((int) (1024 * 1024))
#define ALIGN(x) __attribute__((aligned(x)))
#define VEC_LENGTH 8

// used for storing and displaying execution details
typedef struct stats {
    double exec_time;
    double err;
    double speedup;
    char id[10];
} stats;

// thread arguments
typedef struct thread_data {
    
    float* U;
    float* V;
    uint32_t st, end;
    double (*corr)(float*, float*, uint32_t, uint32_t); // correlation function

} thread_args_t;

// conversion from vector8 to scalar
typedef union vec2scal256 {
    __m256 vec;
    float scal[VEC_LENGTH];
} vec2scal256_t;

// threading vars
pthread_mutex_t sum_mutex;
double total_sum_thr = 0.0f;

// function declarations
double now();
void   init(float** U, float** V);
double corr(float* U, float* V, uint32_t st, uint32_t end);
double vect_corr(float* U, float* V, uint32_t st, uint32_t end);
void * partial_sum(void *args);
void   corrPar(float* U, float* V, uint32_t n, uint32_t n_threads, uint8_t mode);
void   display_results(stats* arr, uint32_t n);
void   finalize(float* U, float* V);
void   run(float* U, float* V, uint32_t n, uint32_t n_threads, stats stats_arr[]);

int main() {
    
    uint8_t i;
    // two input vectors aligened on 32 bytes 
    float* U ALIGN(32);
    float* V ALIGN(32);
    stats stats_arr[4] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f},
                          {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}}, tmp[4];
    uint32_t n_threads = (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
    
    printf("\n[INFO] Vector length: %d", N);
    printf("\n[INFO] Running threads: %d", n_threads);
    
    init(&U, &V);
    
    //warmup run
    run(U, V, 10240, n_threads, tmp);
    printf("\n[INFO] Warmup run finished");
    
    // run the operations T times to get an average estimate of the speedups
    printf("\n[INFO] Executing operations %d times", T);
    for (i=0; i<T; ++i) {
        run(U, V, N, n_threads, stats_arr);
    }
    
    // clear memory
    finalize(U, V);
    
    display_results(stats_arr, 4);
    printf("\n\n");
    
    return 0;
}

double now() {
    
    struct timeval t;
    double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec; f_t = f_t/1000000.0; f_t += t.tv_sec;
    
    return f_t;
}

void init(float** U, float** V) {

    uint32_t i, j;
    int status;
    int align_to = (int) VEC_LENGTH * sizeof(float);
    float *u, *v;
    
    
    // using posix_memalign() because malloc() fails to allocate from an aligned address
    status = posix_memalign((void **)U, align_to, sizeof(float) * N);
    status += posix_memalign((void **)V, align_to, sizeof(float) * N);
    
    if (status == 0) {
        
        srand(0);
        
        u = *U;
        v = *V;
        for (i=0; i<N; ++i) {
            
            u[i] = (float) rand() / (float) RAND_MAX;
            v[i] = (float) rand() /(float) RAND_MAX;
        }
        
        printf("\n[INFO] Init success");
    }
    else {
        printf("\n[ERROR] Init failed.\n\n");
        exit(EXIT_FAILURE);
    }
}

double corr(float* U, float* V, uint32_t st, uint32_t end) {

    uint32_t i;
    double sum = 0.0f;
    
    for (i=st; i<end; ++i)
        sum += (double) (cosf(sqrtf(U[i])) * cosf(sqrtf(V[i])));
    
    return sum;
}

double vect_corr(float* U, float* V, uint32_t st, uint32_t end) {

    uint32_t i, j;
    double sum = 0.0f;
    float* tmp ALIGN(32);
    __m256 vec_sum = _mm256_setzero_ps();
    __m256 vec_mul;
    __m128 half1, half2;
    __m256* vec_U = (__m256*) &U[st];
    __m256* vec_V = (__m256*) &V[st];
    vec2scal256_t arr_U, arr_V;
    uint32_t len = end - st;
    uint32_t n_iters = len / VEC_LENGTH;
    int mod = len % VEC_LENGTH;
    
    for (i=0; i<n_iters; ++i) {
        
        arr_U.vec = _mm256_sqrt_ps(vec_U[i]);
        arr_V.vec = _mm256_sqrt_ps(vec_V[i]);
        
        for(j=0; j<VEC_LENGTH; ++j) {
            arr_U.scal[j] = cosf(arr_U.scal[j]);
            arr_V.scal[j] = cosf(arr_V.scal[j]);
        }
        
        vec_mul = _mm256_mul_ps(arr_U.vec, arr_V.vec);
        vec_sum = _mm256_add_ps(vec_sum, vec_mul);
    }
    
    // calculate for the remaining values that were not executed in the above loop
    if (mod != 0) {
        
        arr_U.vec = _mm256_setzero_ps();
        arr_V.vec = _mm256_setzero_ps();
        for (i=0; i<mod; ++i) {
            arr_U.scal[i] = ((float*) &vec_U[n_iters])[i];
            arr_V.scal[i] = ((float*) &vec_V[n_iters])[i];
        }
        arr_U.vec = _mm256_sqrt_ps(arr_U.vec);
        arr_V.vec = _mm256_sqrt_ps(arr_V.vec);
        
        for(j=0; j<VEC_LENGTH; ++j) {
            arr_U.scal[j] = cosf(arr_U.scal[j]);
            arr_V.scal[j] = cosf(arr_V.scal[j]);
        }
        
        vec_mul = _mm256_mul_ps(arr_U.vec, arr_V.vec);
        vec_sum = _mm256_add_ps(vec_sum, vec_mul);
        
    }
    
    half1 = _mm256_extractf128_ps(vec_sum, 0);
    half2 = _mm256_extractf128_ps(vec_sum, 1);
    half1 = _mm_add_ps(half1, half2);
    
    tmp = (float *) &half1;
    sum = (double) (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    
    return sum;
}

void * partial_sum(void *args) {

    int i;
    double sum = 0.f;
    thread_args_t* tmp = (thread_args_t*) args;
    float* U = tmp->U;
    float* V = tmp->V;
    int st = tmp->st;
    int end = tmp->end;
    
    sum = tmp->corr(tmp->U, tmp->V, tmp->st, tmp->end);
    
    pthread_mutex_lock(&sum_mutex);
    total_sum_thr += sum;
    pthread_mutex_unlock(&sum_mutex);

    pthread_exit(NULL);
}

void corrPar(float* U, float* V, uint32_t n, uint32_t n_threads, uint8_t mode) {
    
    uint32_t i;
    pthread_t tid[n_threads];
    thread_args_t thread_args[n_threads];
    double total_sum = 0.0;
    uint32_t len = (n / n_threads);
    
    total_sum_thr = 0.0f;
    
    if (mode == 1)
        len += len % VEC_LENGTH;
    
    for (i=0; i<n_threads; ++i) {
   
        thread_args[i].corr = (mode == 0) ? corr : vect_corr;
        thread_args[i].U = U;
        thread_args[i].V = V;
        thread_args[i].st = (int) i * len;
        
        // intermediate addresses might not be aligned depending on the number of threads and size of a vector
        // therefore we make sure we pass the addresses that are aligned to 32 bytes.
        if (mode == 1 && !is_aligned(&U[thread_args[i].st], 32)) {
            while(!is_aligned(&U[++thread_args[i].st], 32));
        }
        thread_args[i].end = (int) (i != n_threads-1) ? (thread_args[i].st + len) : n;
        
        if (pthread_create(&tid[i], NULL, partial_sum, &thread_args[i]) != 0) {
            printf("\n[ERROR] Failed to create the thread %d\n\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    for (i=0; i<n_threads; ++i)
        pthread_join(tid[i], NULL);
}

void display_results(stats* arr, uint32_t n) {
    
    uint32_t i;
    double inv_T = 1.f / (double) T;
    
    printf("\n\nExecution details:");
    printf("\n\noperation\texec time (msec)\tspeed up\tsum error");
    printf("\n==========================================================================");
    for (i=0; i<n; ++i) {
        printf("\n%s\t%.3f\t\t\t%.2f\t\t%e", arr[i].id, 1e3 * arr[i].exec_time * inv_T, arr[i].speedup * inv_T, arr[i].err * inv_T);
    }
    printf("\n\n");
}

void finalize(float* U, float* V) {
    
     if (U) free(U);
     if (V) free(V);
}

void run(float* U, float* V, uint32_t n, uint32_t n_threads, stats stats_arr[]) {
    
    // local variables for calculations
    double sum_scal, sum_vec;
    double start, end;
    
    // scalar reduction
    start = now();
    sum_scal = corr(U, V, 0, n);
    end = now();
    stats_arr[0].exec_time += (double) (end - start);
    stats_arr[0].speedup += 1.f;
    stats_arr[0].err += 0.f;
    strcpy(stats_arr[0].id, "scalar seq");
    
    // vector reduction
    start = now();
    sum_vec = vect_corr(U, V, 0, n);
    end = now();
    stats_arr[1].exec_time += (double) (end - start);
    stats_arr[1].speedup += stats_arr[0].exec_time / stats_arr[1].exec_time;
    stats_arr[1].err += fabs(sum_scal - sum_vec) / sum_scal;
    strcpy(stats_arr[1].id, "vector seq");
    
    // scalar thread reduction (mode=0)
    start = now();
    corrPar(U, V, n, n_threads, 0);
    end = now();
    stats_arr[2].exec_time += (double) (end - start);
    stats_arr[2].speedup += stats_arr[0].exec_time / stats_arr[2].exec_time;
    stats_arr[2].err += fabs(sum_scal - total_sum_thr) / sum_scal;
    strcpy(stats_arr[2].id, "scalar thr");
    
    // vector thread reduction (mode=1)
    start = now();
    corrPar(U, V, n, n_threads, 1);
    end = now();
    stats_arr[3].exec_time += (double) (end - start);
    stats_arr[3].speedup += stats_arr[0].exec_time / stats_arr[3].exec_time;
    stats_arr[3].err += fabs(sum_scal - total_sum_thr) / sum_scal;
    strcpy(stats_arr[3].id, "vector thr");
}