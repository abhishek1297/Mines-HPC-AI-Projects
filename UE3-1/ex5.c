/*
* Author: Abhishek Purandare
* Course: UE3 High-Performance Computing
*
* The file contains exercise 5
*
*/
/*
The following code is same as the ex1to4.c file with additions of error handling with respect to the negative values.

The approach is,
1. declare global variables,
negative_found: set when a negative value is encountered
cond_mutex: used for accessing the above variable as well as used for signaling
neg_val: used for signaling and listening
n_finished: number of threads that are finished.

2. Use pthread_cond_wait and pthread_cond_signal to notify when negative value is encountered.
3. Create an additional thread that waits until some thread sends a signal
4. For each computational thread check if the sum is -1. If yes, then lock the cond_mutex. Set the negative_found and send the signal, and exit.
5. For each computational thread check if the sum > 0. If yes, then add the sum to the global sum and increment n_finished. If n_finished == n_threads then we send the signal. Otherwise, the thread_killer will wait indefinitely.
6. Running thread_killer will go into wait state until a signal is received. If so, it will execute pthread_cancel() to cancel all threads that are currently running.
7. Once the control goes back to the main thread. We then handle the exception as per needed.
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
#define N ((int) (1024 * 1024))
#define T 1
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
    uint32_t st, end, n_threads;
    double (*corr)(float*, float*, uint32_t, uint32_t); // correlation function

} thread_args_t;

// thread killer arguments
typedef struct thread_killer_data {
    uint32_t n_threads;
    pthread_t* tid;
} thread_killer_args_t;

// conversion from vector8 to scalar
typedef union vec2scal256 {
    __m256 vec;
    float scal[VEC_LENGTH];
} vec2scal256_t;

// threading vars
pthread_mutex_t sum_mutex, cond_mutex;
double total_sum_thr = 0.0f;
pthread_cond_t neg_val;
uint8_t negative_found = 0, n_finished = 0;

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
    
    // all 4 implementations reset the value of negative_found before running.
    run(U, V, N, n_threads, stats_arr);
    
    // clear memory
    finalize(U, V);
    
    if (!negative_found)
        display_results(stats_arr, 4);
    else
        printf("\n[ERROR] Negative value(s) encoutered by one of the threads");
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
        
        //inject 10 negative values at random places in vector U
        for (i=0; i<10; ++i) {
            j = (int) (rand() % (N + 1));
            u[j] = -1.f;
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
    
    for (i=st; i<end; ++i) {
        if (U[i] < 0 || V[i] < 0)
            return -1.f;

        sum += (double) (cosf(sqrtf(U[i])) * cosf(sqrtf(V[i])));
    }
    
    return sum;
}

double vect_corr(float* U, float* V, uint32_t st, uint32_t end) {
    
    uint32_t i, j;
    int mask1, mask2;
    double sum = 0.0f;
    float* tmp ALIGN(32);
    const __m256 all_zeros = _mm256_setzero_ps();
    __m256 vcmp1, vcmp2;
    __m256 vec_sum = all_zeros;
    __m256 vec_mul;
    __m128 half1, half2;
    __m256* vec_U = (__m256*) &U[st];
    __m256* vec_V = (__m256*) &V[st];
    vec2scal256_t arr_U, arr_V;
    uint32_t len = end - st;
    uint32_t n_iters = len / VEC_LENGTH;
    int mod = len % VEC_LENGTH;
    
    for (i=0; i<n_iters; ++i) {
        
        vcmp1 = _mm256_cmp_ps(vec_U[i], all_zeros, _CMP_LT_OS);
        vcmp2 = _mm256_cmp_ps(vec_V[i], all_zeros, _CMP_LT_OS);
        mask1 = _mm256_movemask_ps(vcmp1);
        mask2 = _mm256_movemask_ps(vcmp2);
        if (mask1 != 0 || mask2 != 0)
            return -1.f;
        
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
                
        vcmp1 = _mm256_cmp_ps(arr_U.vec, all_zeros, _CMP_LT_OS);
        vcmp2 = _mm256_cmp_ps(arr_V.vec, all_zeros, _CMP_LT_OS);
        mask1 = _mm256_movemask_ps(vcmp1);
        mask2 = _mm256_movemask_ps(vcmp2);
        if (mask1 != 0 || mask2 != 0)
            return -1.f;
        
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

void* thread_killer(void* args) {
    
    int i;
    thread_killer_args_t* tmp = (thread_killer_args_t*) args;
    uint32_t n_threads = tmp->n_threads;
    pthread_t* tid = (pthread_t*) tmp->tid;
    
    pthread_mutex_lock(&cond_mutex);
    while (!negative_found && n_finished != n_threads)
        pthread_cond_wait(&neg_val, &cond_mutex);
    pthread_mutex_unlock(&cond_mutex);
    
    if (n_finished != n_threads) {
        for (i=0; i<n_threads; ++i)
            pthread_cancel(tid[i]);
    }
    
    pthread_exit(NULL);
}

void* partial_sum(void* args) {

    int i;
    double sum = 0.f;
    thread_args_t* tmp = (thread_args_t*) args;
    float* U = tmp->U;
    float* V = tmp->V;
    uint32_t st = tmp->st;
    uint32_t end = tmp->end;
    uint32_t n_threads = tmp->n_threads;
    
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
    
    sum = tmp->corr(tmp->U, tmp->V, tmp->st, tmp->end);
    
    pthread_mutex_lock(&cond_mutex);
    if (sum < 0) {
        negative_found = 1;
        pthread_cond_signal(&neg_val);
    }
    else {
        pthread_mutex_lock(&sum_mutex);
        total_sum_thr += sum;
        pthread_mutex_unlock(&sum_mutex);
    }
    
    if (++n_finished == n_threads)
        pthread_cond_signal(&neg_val);
    pthread_mutex_unlock(&cond_mutex);
    
    pthread_exit(NULL);
}

void corrPar(float* U, float* V, uint32_t n, uint32_t n_threads, uint8_t mode) {
    
    uint32_t i;
    pthread_attr_t attr;
    pthread_t tid[n_threads], t_killer;
    thread_args_t thread_args[n_threads];
    thread_killer_args_t killer_args;

    double total_sum = 0.0;
    uint32_t len = (n / n_threads);
    total_sum_thr = 0.0f;
    negative_found = 0;
    n_finished = 0;
    
    pthread_mutex_init(&cond_mutex, NULL);
    pthread_cond_init(&neg_val, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    killer_args.tid = (pthread_t*) tid;
    killer_args.n_threads = n_threads;
    
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
        thread_args[i].n_threads = n_threads;
        
        if (pthread_create(&tid[i], &attr, partial_sum, &thread_args[i]) != 0) {
            printf("\n[ERROR] Failed to create the thread %d\n\n", i);
            exit(EXIT_FAILURE);
        }
    }
    pthread_create(&t_killer, &attr, thread_killer, &killer_args);
    pthread_join(t_killer, NULL);
    
    for (i=0; i<n_threads; ++i)
        pthread_join(tid[i], NULL);
    
    
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&cond_mutex);
    pthread_cond_destroy(&neg_val);
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
    negative_found = 0;
    start = now();
    sum_scal = corr(U, V, 0, n);
    end = now();
    stats_arr[0].exec_time += (double) (end - start);
    stats_arr[0].speedup += 1.f;
    stats_arr[0].err += 0.f;
    strcpy(stats_arr[0].id, "scalar seq");
    negative_found = (int) (sum_scal < 0);
    
    // vector reduction
    negative_found = 0;
    start = now();
    sum_vec = vect_corr(U, V, 0, n);
    end = now();
    stats_arr[1].exec_time += (double) (end - start);
    stats_arr[1].speedup += stats_arr[0].exec_time / stats_arr[1].exec_time;
    stats_arr[1].err += fabs(sum_scal - sum_vec) / sum_scal;
    strcpy(stats_arr[1].id, "vector seq");
    negative_found = (int) (sum_vec < 0);
    
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
