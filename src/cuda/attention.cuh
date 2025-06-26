#include <math_constants.h>
#include <cuda/std/cstdint>

template <typename T>
struct kv_cache {
    T *k;
    T *v;
    __device__ kv_cache(T *k_, T *v_) : k(k_), v(v_) {}
};

template <typename T>
__device__ kv_cache<T> locate_cache(
    T *const *pages,
    long long sbuf,      // sequence stride
    long long skv,       //   k to v stride
    long long sh,        //  kv head stride
    unsigned long long const bs, // context tile
    unsigned long long const head,
    unsigned long long const pos) {
    sh *= head;
    sbuf *= pos % bs;
    uint8_t *page = (uint8_t *)pages[pos / bs];
    return kv_cache<T>((T *)(page + sh + sbuf), (T *)(page + sh + sbuf + skv));
}

template <typename T>
__device__ void __attn(
    T *const *kv_pages,
    T const *q_,           // n@ x d
    T *o_,                 // n@ x d
    bool const *mask_,     // n x s
    T *m,                  // s
    T *l,                  // s
    unsigned long long const n,      // sequence length
    unsigned long long const d,      // head dim
    unsigned long long const ts,     // = s/bs
    unsigned long long const bs,     // context tile
    long long const sq,      //        q stride
    long long const so,      //        o stride
    long long const kv_sbuf, // sequence stride
    long long const kv_skv,  //   k to v stride
    long long const kv_sh,   //  kv head stride
    float const scale) {
    // (batch x head) x (bn)
    unsigned long long const head = blockIdx.x;
    unsigned long long const bn = blockDim.x;
    unsigned long long const it = threadIdx.x;
    unsigned long long const tn = (n + bn - 1) / bn;

    extern __shared__ T sram[];
    int tile_size = bs * d;
    T *qi = sram;
    T *kj = &sram[tile_size];
    T *vj = &sram[tile_size * 2];
    T *x = &sram[tile_size * 3];
    // kv
    for (unsigned long long ikvb = 0; ikvb < ts; ++ikvb) {
        // 加载kv
        { // 每个线程拷贝 k/v 的一行，拷贝整个 kv block 到 local memory
            unsigned long long const end = (ikvb + 1) * bs;
            for (unsigned long long ikv = ikvb * bs + it, i = it; ikv < end; ikv += bn, i += bn) {
                kv_cache const cache = locate_cache(kv_pages, kv_sbuf, kv_skv, kv_sh, bs, head, ikv);
                for (unsigned long long j = 0; j < d; ++j) {
                    kj[i * d + j] = cache.k[j];
                    vj[i * d + j] =cache.v[j];
                }
            }
            __syncthreads();
        }
        { // 每个线程计算 q 的一行

            for (unsigned long long iqb = 0; iqb < tn; ++iqb) {
                unsigned long long iq = iqb * bn + it;
                if (iq >= n) {
                    break;
                }
                // locate data 加载q
                T const *q = q_ + iq * sq;
                T *o = o_ + iq * so;
                bool const *mask = mask_ + iq * n + ikvb * bs;
                // load data
                for (unsigned long long i = 0; i < d; ++i) {
                    qi[i] = q[i];
                }

                T const mi_1 = m[iq];
                T const di_1 = l[iq];

                // score = q @ k^T / √d
                T mi = mi_1;
                for (unsigned long long i = 0; i < bs; ++i) {
                    if (!mask[i]) {
                        x[i] = -CUDART_INF_F;
                    } else {
                        T const *k = kj + i * d;

                        for (unsigned long long j = 0; j < d; ++j) {
                            x[i] += qi[j] * kj[j];
                        }
                        x[i] *= scale;

                        if (x[i] > mi) {
                            mi = x[i];
                        }
                    }
                }
                // P = exp(S - row_m), row_l = rowsum(P)
                T sum = 0;
                for (unsigned long long i = 0; i < bs; ++i) {
                    x[i] = std::exp(x[i] - mi);
                    sum += x[i];
                }

                T exp = di_1 * std::exp(mi_1 - mi);
                T di = exp + sum;
                // 更新mi,di
                m[iq] = mi;
                l[iq] = di;

                T rdi = 1 / di;
                exp *= rdi;
                for (unsigned long long i = 0; i < bs; ++i) {
                    x[i] *= rdi;
                }
            }
            __syncthreads();
        }
    }
}
__global__ void __attn_f64(
    double *const *kv_pages,
    double const *q_,
    double *o_,
    bool const *mask_,
    double *m,
    double *l,
    unsigned long long const n,
    unsigned long long const d,
    unsigned long long const ts,
    unsigned long long const bs,
    long long const sq,
    long long const so,
    long long const kv_sbuf,
    long long const kv_skv,
    long long const kv_sh,
    float const scale) {
    // 调用模板实现
    __attn<double>(kv_pages, q_, o_, mask_, m, l, n, d, ts, bs, sq, so, kv_sbuf, kv_skv, kv_sh, scale);
}
