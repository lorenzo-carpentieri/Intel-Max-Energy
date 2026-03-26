#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <synergy.hpp>

#include "../include/utils/utils.hpp"

namespace {

constexpr int DEFAULT_SIZE = 512 * 512;
constexpr int DEFAULT_ITERS = 1000;
constexpr int DEFAULT_RUNS = 5;
constexpr std::size_t DEFAULT_LOCAL_SIZE = 256;

std::string csv_path;
synergy::frequency freq = 1600;

std::string build_csv_file_path(const std::string& csv_dir) {
    if (csv_dir.empty()) {
        return {};
    }
    return csv_dir + "/ftle_" + std::to_string(freq) + ".csv";
}

sycl::float4 float4mul(const sycl::float4& a, const sycl::float4& b) {
    sycl::float4 c;
    c.x() = a.x() * b.x() + a.y() * b.y();
    c.y() = a.x() * b.y() + a.y() * b.w();
    c.z() = a.z() * b.x() + a.w() * b.z();
    c.w() = a.z() * b.y() + a.w() * b.w();
    return c;
}

sycl::float4 float4trp(const sycl::float4& a) {
    sycl::float4 b = a;
    const float y = b.y();
    b.y() = b.z();
    b.z() = y;
    return b;
}

sycl::float4 float4symm(const sycl::float4& a) {
    return (float4trp(a) + a) * 0.5f;
}

sycl::float2 float4invariants(const sycl::float4& m) {
    sycl::float2 pqr;
    pqr.x() = m.x() * m.w() - m.y() * m.z();
    pqr.y() = -(m.x() + m.w());
    return pqr;
}

sycl::float2 float2squareroots(const sycl::float2& a) {
    const float discrim = a.y() * a.y() - 4.0f * a.x();
    sycl::float2 b;

    if (discrim >= 0.0f) {
        const float root = sycl::sqrt(discrim);
        b.x() = (-a.y() - root) / 2.0f;
        b.y() = (-a.y() + root) / 2.0f;
    } else {
        const float root = sycl::sqrt(-discrim);
        b.x() = -a.x() / 2.0f;
        b.y() = root / 2.0f;
    }

    return b;
}

sycl::float2 float4eigenvalues(const sycl::float4& m) {
    return float2squareroots(float4invariants(m));
}

class FtleSimulation {
  public:
    FtleSimulation()
        : requested_size_(DEFAULT_SIZE),
          num_iters_(DEFAULT_ITERS),
          num_runs_(DEFAULT_RUNS),
          local_size_(DEFAULT_LOCAL_SIZE),
          cell_size_(0.1f, 0.1f),
          advection_time_(0.5f) {}

    void SetProblemSize(std::size_t size) { requested_size_ = size; }
    void SetNumIters(int iters) { num_iters_ = std::max(1, iters); }
    void SetNumRuns(int runs) { num_runs_ = std::max(1, runs); }
    void SetLocalSize(std::size_t local_size) {
        local_size_ = std::max<std::size_t>(1, local_size);
    }

    void Start() {
        const int width = std::max(1, static_cast<int>(std::floor(std::sqrt(requested_size_))));
        const std::size_t problem_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(width);

        std::vector<sycl::float2> flow_map(problem_size);
        std::vector<float> output(problem_size, 0.0f);
        init_flow_map(flow_map);

        int dev_id = 0;
        synergy::frequency uncore_freq = 0;
        sycl::device device = utils::device::pick_device(dev_id);
        synergy::queue q(uncore_freq, freq, device);

        synergy::utils::check_core_freq(q.get_synergy_device(), freq, 1000);

        const std::size_t global_size =
            ((problem_size + local_size_ - 1) / local_size_) * local_size_;
        const sycl::float2 cell_size = cell_size_;
        const float advection_time = advection_time_;

        sycl::float2* flow_map_d = sycl::malloc_device<sycl::float2>(problem_size, q);
        float* output_d = sycl::malloc_device<float>(problem_size, q);

        if (flow_map_d == nullptr || output_d == nullptr) {
            if (flow_map_d != nullptr) {
                sycl::free(flow_map_d, q);
            }
            if (output_d != nullptr) {
                sycl::free(output_d, q);
            }
            throw std::runtime_error("Device allocation failed for FTLE");
        }

        q.memcpy(flow_map_d, flow_map.data(), problem_size * sizeof(sycl::float2)).wait();
        q.memcpy(output_d, output.data(), problem_size * sizeof(float)).wait();

        std::vector<synergy::device> synergy_devs;
        synergy_devs.push_back(q.get_synergy_device());
        synergy::profiler::PowerProfiler power_prof(synergy_devs, 10);

        std::cout << "[INFO] Device: "
                  << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "[INFO] FTLE grid width: " << width
                  << ", elements: " << problem_size
                  << ", iterations: " << num_iters_
                  << ", runs: " << num_runs_
                  << ", local_size: " << local_size_ << '\n';

        for (int run_id = 0; run_id < num_runs_; ++run_id) {
            power_prof.start();
            utils::data_types::energy_t start_energy = q.device_energy_consumption();
            auto time_start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters_; ++iter) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<class FtleStandaloneKernel>(
                        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size_)),
                        [=](sycl::nd_item<1> item) {
                            const std::size_t gid = item.get_global_linear_id();
                            if (gid >= problem_size) {
                                return;
                            }

                            const int tx = static_cast<int>(gid % static_cast<std::size_t>(width));
                            const int ty = static_cast<int>(gid / static_cast<std::size_t>(width));

                            if (tx < 1 || tx >= width - 1 || ty < 1 || ty >= width - 1) {
                                output_d[gid] = 0.0f;
                                return;
                            }

                            const sycl::float2 left = flow_map_d[gid - 1];
                            const sycl::float2 right = flow_map_d[gid + 1];
                            const sycl::float2 top = flow_map_d[gid - width];
                            const sycl::float2 bottom = flow_map_d[gid + width];

                            sycl::float2 delta2;
                            delta2.x() = 2.0f * cell_size.x();
                            delta2.y() = 2.0f * cell_size.y();

                            sycl::float4 jacobi;
                            jacobi.x() = (right.x() - left.x()) / delta2.x();
                            jacobi.y() = (bottom.x() - top.x()) / delta2.y();
                            jacobi.z() = (right.y() - left.y()) / delta2.x();
                            jacobi.w() = (bottom.y() - top.y()) / delta2.y();

                            const sycl::float4 jacobi_t = float4trp(jacobi);
                            const sycl::float4 cauchy = float4mul(jacobi_t, jacobi);
                            const sycl::float4 cauchy_symm = float4symm(cauchy);

                            sycl::float2 eigenvalues = float4eigenvalues(cauchy_symm);
                            eigenvalues += float4eigenvalues(cauchy_symm);
                            eigenvalues += float4eigenvalues(cauchy_symm);
                            eigenvalues += float4eigenvalues(cauchy_symm);

                            const float max_eigenvalue =
                                sycl::max(eigenvalues.x(), eigenvalues.y());
                            output_d[gid] =
                                sycl::log(sycl::sqrt(max_eigenvalue)) / sycl::fabs(advection_time);
                        });
                });
            }

            q.wait();

            auto time_end = std::chrono::high_resolution_clock::now();
            utils::data_types::energy_t end_energy = q.device_energy_consumption();
            power_prof.stop();

            const double time_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
            const utils::data_types::energy_t total_dev_energy = end_energy - start_energy;

            sycl::event copy_event =
                q.memcpy(output.data(), output_d, problem_size * sizeof(float));
            copy_event.wait();

            sycl::float2 center_sample = flow_map[problem_size / 2];
            std::cout << "[INFO] Run " << run_id
                      << ": time_ms=" << time_ms
                      << ", device_energy_mj=" << total_dev_energy
                      << ", sample_output=" << output[problem_size / 2]
                      << ", sample_flow=(" << center_sample.x()
                      << "," << center_sample.y() << ")\n";

            if (!csv_path.empty()) {
                utils::logger::Logger log(build_csv_file_path(csv_path));
                synergy::power_trace_t power_trace = power_prof.get_power_execution_data()[0];
                synergy::freq_trace_t freq_trace = power_prof.get_freq_execution_data()[0];
                synergy::temperature_trace_t temp_trace =
                    power_prof.get_temperature_execution_data()[0];
                temp_trace.clear();

                utils::logger::Logger::ProfilingInfo<float> prof_info{
                    "h",
                    "ftle",
                    problem_size,
                    num_iters_,
                    num_runs_,
                    run_id,
                    time_ms,
                    total_dev_energy,
                    0,
                    power_trace,
                    freq_trace,
                    temp_trace,
                    utils::data_types::GPUMode::FLAT};
                log.log_result(prof_info);
            }

            power_prof.clean();
        }

        sycl::free(flow_map_d, q);
        sycl::free(output_d, q);
    }

  private:
    std::size_t requested_size_;
    int num_iters_;
    int num_runs_;
    std::size_t local_size_;
    sycl::float2 cell_size_;
    float advection_time_;

    static void init_flow_map(std::vector<sycl::float2>& flow_map) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto& value : flow_map) {
            value.x() = dist(gen);
            value.y() = dist(gen);
        }
    }
};

void print_usage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name << " <size> <iters> <csv_dir> <freq> [runs] [local_size]\n"
              << "  " << exe_name
              << " --size=<n> --iters=<n> [--csv-path=<dir>] [--freq=<mhz>] [--runs=<n>] [--local-size=<n>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    FtleSimulation sim;

    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0) {
            sim.SetProblemSize(static_cast<std::size_t>(std::stoull(argv[1])));
            if (argc >= 3) {
                sim.SetNumIters(std::stoi(argv[2]));
            }
            if (argc >= 4) {
                csv_path = argv[3];
            }
            if (argc >= 5) {
                freq = static_cast<synergy::frequency>(std::stoul(argv[4]));
            }
            if (argc >= 6) {
                sim.SetNumRuns(std::stoi(argv[5]));
            }
            if (argc >= 7) {
                sim.SetLocalSize(static_cast<std::size_t>(std::stoull(argv[6])));
            }
        } else {
            utils::arg_parse::map_t params = utils::arg_parse::parse(argc, argv);
            for (const auto& [key, value] : params) {
                if (key == "size") {
                    sim.SetProblemSize(static_cast<std::size_t>(std::stoull(value)));
                } else if (key == "iters") {
                    sim.SetNumIters(std::stoi(value));
                } else if (key == "runs") {
                    sim.SetNumRuns(std::stoi(value));
                } else if (key == "csv-path") {
                    csv_path = value;
                } else if (key == "freq") {
                    freq = static_cast<synergy::frequency>(std::stoul(value));
                } else if (key == "local-size") {
                    sim.SetLocalSize(static_cast<std::size_t>(std::stoull(value)));
                } else {
                    throw std::runtime_error("Unknown parameter: " + key);
                }
            }
        }

        sim.Start();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
