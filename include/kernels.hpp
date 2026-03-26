#pragma once
#ifndef REPEATS 
#define REPEATS 512*512
#endif
#ifndef STRIDE 
#define STRIDE 2048
#endif
#define PRIME 1315423911

namespace kernels{
    template <typename T, int ITERS, size_t MEM_ITERS = 1>
    class AIKernel {
    public:
        AIKernel(T* data, const size_t* random_offsets, size_t n)
            : data_(data), random_offsets_(random_offsets), n_(n) {}

        void operator()(sycl::id<1> idx) const {
            size_t i = idx[0];
            if (i >= n_) return;
            float x = 0;
            for (int r = 0; r < REPEATS; ++r) {
                size_t j = ((i ^ r) * PRIME + random_offsets_[r]) % n_;
                data_[i] = data_[j];
                x = data_[i];
                for (int k = 0; k < ITERS; ++k) {
                    x = sycl::fma(x, x, x);
                }

            }

            data_[i] = x;
        }

    private:
        T* data_;          
        const size_t* random_offsets_;
        size_t n_;
    };

    template <typename T>
    class ComputeKernel {
    public:
        ComputeKernel(T* data, std::size_t n, int iters)
            : data_(data), n_(n), iters_(iters) {}

        void operator()(sycl::id<1> idx) const {
            std::size_t i = idx[0];
            if (i >= n_) return;

            float x = data_[i];

            // Heavy arithmetic loop → compute-bound
            for (int k = 0; k < iters_; ++k) {
                x = sycl::sin(x) * sycl::cos(x) + sycl::sqrt(x + 1.0f);
                x = sycl::fma(x, 1.000001f, 0.999999f);
                x = sycl::fma(x, 1.000002f, 0.999998f);
                x = sycl::fma(x, 1.000003f, 0.999997f);
                x = sycl::fma(x, 1.000004f, 0.36f);
                x = sycl::fma(x, 1.000004f, 0.49996f);
                x = sycl::fma(x, 1.2f, 0.599996f);
                x = sycl::fma(x, 1.3f, 0.799996f);
                x = sycl::fma(x, 1.1f, 0.299996f);


            }

            data_[i] = x;
        }

    private:
        T* data_;          // USM pointer
        std::size_t n_;
        int iters_;             // Controls compute intensity
    };
    
    template <typename T>
    class MemoryBoundKernel {
    public:
        MemoryBoundKernel(T* data,
                          std::size_t n,
                          int iters,
                          std::size_t stride = 128)
            : data_(data),
              indices_(nullptr),
              index_count_(0),
              n_(n),
              iters_(iters),
              stride_(stride) {}

        MemoryBoundKernel(T* data,
                          const std::size_t* indices,
                          std::size_t index_count,
                          std::size_t n,
                          int iters,
                          std::size_t stride = 1)
            : data_(data),
              indices_(indices),
              index_count_(index_count),
              n_(n),
              iters_(iters),
              stride_(stride) {}

        void operator()(sycl::id<1> idx) const {
            std::size_t i = idx[0];
            if (i >= n_) return;

            T acc = data_[i];
            std::size_t cursor = i;

            for (int k = 0; k < iters_; ++k) {
                std::size_t j = 0;

                if (indices_ != nullptr && index_count_ != 0) {
                    const std::size_t offset =
                        indices_[static_cast<std::size_t>(k) % index_count_];
                    cursor = (cursor + offset + stride_) % n_;
                    j = cursor;
                } else {
                    j = (i + static_cast<std::size_t>(k) * stride_) % n_;
                }

                acc += data_[j];
            }

            data_[i] = acc;
        }

    private:
        T* data_;                 // USM pointer
        const std::size_t* indices_;
        std::size_t index_count_;
        std::size_t n_;
        int iters_;               // Controls memory traffic
        std::size_t stride_;      // Controls cache behavior
    };


     template <typename T>
    class EmptyKernel {
    public:
        EmptyKernel(){}

        void operator()(sycl::id<1> idx) const {
        }

   
    };
}
