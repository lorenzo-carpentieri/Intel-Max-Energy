#include <sycl/sycl.hpp>

namespace utils{
    namespace profiling{
        double get_kernel_time_ms(sycl::event& e) {
            // Wait for the event to finish
            e.wait();

            // Get start and end timestamps (in nanoseconds)
            uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();

            // Convert nanoseconds to milliseconds
            double duration_ms = static_cast<double>(end - start) * 1e-6;
            return duration_ms;
        }
    }
}