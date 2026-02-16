#pragma once
namespace kernels{
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
                x = sycl::fma(x, 1.000004f, 0.999996f);
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
            : data_(data), n_(n), iters_(iters), stride_(stride) {}

        void operator()(sycl::id<1> idx) const {
            std::size_t i = idx[0];
            if (i >= n_) return;

            T tmp = 0;

            // Memory-heavy loop: many global loads
            for (int k = 0; k < iters_; ++k) {
                std::size_t j = (i + k * stride_) % n_;
                tmp += data_[j];
            }

            // Global store
            data_[i] = tmp;
        }

    private:
        T* data_;            // USM pointer
        std::size_t n_;
        int iters_;          // Controls memory traffic
        std::size_t stride_; // Controls cache behavior
    };


     template <typename T>
    class EmptyKernel {
    public:
        EmptyKernel(){}

        void operator()(sycl::id<1> idx) const {
        }

   
    };
}