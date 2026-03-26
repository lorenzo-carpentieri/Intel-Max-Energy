#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <synergy.hpp>

#include "../include/utils/utils.hpp"

namespace {

constexpr std::size_t DEFAULT_SIZE = 1 << 20;
constexpr int DEFAULT_ITERS = 100;
constexpr int DEFAULT_RUNS = 5;
constexpr std::size_t DEFAULT_LOCAL_SIZE = 256;
constexpr int DEFAULT_NEIGH_COUNT = 15;

std::string csv_path;
synergy::frequency freq = 1600;

std::string build_csv_file_path(const std::string& csv_dir) {
    if (csv_dir.empty()) {
        return {};
    }
    return csv_dir + "/mol_dyn_" + std::to_string(freq) + ".csv";
}

class MolDynSimulation {
  public:
    MolDynSimulation()
        : problem_size_(DEFAULT_SIZE),
          num_iters_(DEFAULT_ITERS),
          num_runs_(DEFAULT_RUNS),
          local_size_(DEFAULT_LOCAL_SIZE),
          neigh_count_(DEFAULT_NEIGH_COUNT),
          cutsq_(50.0f),
          lj1_(20.0f),
          lj2_(0.003f) {}

    void SetProblemSize(std::size_t size) { problem_size_ = std::max<std::size_t>(1, size); }
    void SetNumIters(int iters) { num_iters_ = std::max(1, iters); }
    void SetNumRuns(int runs) { num_runs_ = std::max(1, runs); }
    void SetLocalSize(std::size_t local_size) {
        local_size_ = std::max<std::size_t>(1, local_size);
    }
    void SetNeighborCount(int neigh_count) { neigh_count_ = std::max(1, neigh_count); }

    void Start() {
        std::vector<sycl::float4> input(problem_size_);
        std::vector<sycl::float4> output(problem_size_, sycl::float4(0.0f));
        std::vector<int> neighbors(static_cast<std::size_t>(neigh_count_) * problem_size_);
        init_data(input, neighbors);

        int dev_id = 0;
        synergy::frequency uncore_freq = 0;
        sycl::device device = utils::device::pick_device(dev_id);
        synergy::queue q(uncore_freq, freq, device);

        synergy::utils::check_core_freq(q.get_synergy_device(), freq, 1000);

        const std::size_t global_size =
            ((problem_size_ + local_size_ - 1) / local_size_) * local_size_;
        const std::size_t problem_size = problem_size_;
        const int neigh_count = neigh_count_;
        const float cutsq = cutsq_;
        const float lj1 = lj1_;
        const float lj2 = lj2_;

        sycl::float4* input_d = sycl::malloc_device<sycl::float4>(problem_size, q);
        sycl::float4* output_d = sycl::malloc_device<sycl::float4>(problem_size, q);
        int* neighbors_d = sycl::malloc_device<int>(neighbors.size(), q);

        if (input_d == nullptr || output_d == nullptr || neighbors_d == nullptr) {
            if (input_d != nullptr) {
                sycl::free(input_d, q);
            }
            if (output_d != nullptr) {
                sycl::free(output_d, q);
            }
            if (neighbors_d != nullptr) {
                sycl::free(neighbors_d, q);
            }
            throw std::runtime_error("Device allocation failed for mol_dyn");
        }

        q.memcpy(input_d, input.data(), problem_size * sizeof(sycl::float4)).wait();
        q.memcpy(output_d, output.data(), problem_size * sizeof(sycl::float4)).wait();
        q.memcpy(neighbors_d, neighbors.data(), neighbors.size() * sizeof(int)).wait();

        std::vector<synergy::device> synergy_devs;
        synergy_devs.push_back(q.get_synergy_device());
        synergy::profiler::PowerProfiler power_prof(synergy_devs, 10);

        std::cout << "[INFO] Device: "
                  << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "[INFO] mol_dyn problem_size: " << problem_size
                  << ", neigh_count: " << neigh_count
                  << ", iterations: " << num_iters_
                  << ", runs: " << num_runs_
                  << ", local_size: " << local_size_ << '\n';

        for (int run_id = 0; run_id < num_runs_; ++run_id) {
            power_prof.start();
            utils::data_types::energy_t start_energy = q.device_energy_consumption();
            auto time_start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters_; ++iter) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<class MolDynStandaloneKernel>(
                        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size_)),
                        [=](sycl::nd_item<1> item) {
                            const std::size_t gid = item.get_global_linear_id();
                            if (gid >= problem_size) {
                                return;
                            }

                            const sycl::float4 ipos = input_d[gid];
                            sycl::float4 force(0.0f);

                            for (int j = 0; j < neigh_count; ++j) {
                                const std::size_t neighbor_index =
                                    static_cast<std::size_t>(j) * problem_size + gid;
                                const int jidx = neighbors_d[neighbor_index];
                                const sycl::float4 jpos = input_d[jidx];

                                const float delx = ipos.x() - jpos.x();
                                const float dely = ipos.y() - jpos.y();
                                const float delz = ipos.z() - jpos.z();
                                float r2inv = delx * delx + dely * dely + delz * delz;

                                if (r2inv > 0.0f && r2inv < cutsq) {
                                    r2inv = 10.0f / r2inv;
                                    const float r6inv = r2inv * r2inv * r2inv;
                                    const float force_c = r2inv * r6inv * (lj1 * r6inv - lj2);

                                    force.x() += delx * force_c;
                                    force.y() += dely * force_c;
                                    force.z() += delz * force_c;
                                }
                            }

                            output_d[gid] = force;
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

            q.memcpy(output.data(), output_d, problem_size * sizeof(sycl::float4)).wait();

            const std::size_t sample_idx = problem_size / 2;
            const sycl::float4 sample_force = output[sample_idx];
            std::cout << "[INFO] Run " << run_id
                      << ": time_ms=" << time_ms
                      << ", device_energy_mj=" << total_dev_energy
                      << ", sample_force=(" << sample_force.x()
                      << "," << sample_force.y()
                      << "," << sample_force.z() << ")\n";

            if (!csv_path.empty()) {
                utils::logger::Logger log(build_csv_file_path(csv_path));
                synergy::power_trace_t power_trace = power_prof.get_power_execution_data()[0];
                synergy::freq_trace_t freq_trace = power_prof.get_freq_execution_data()[0];
                synergy::temperature_trace_t temp_trace =
                    power_prof.get_temperature_execution_data()[0];
                temp_trace.clear();

                utils::logger::Logger::ProfilingInfo<float> prof_info{
                    "h",
                    "mol_dyn",
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

        sycl::free(input_d, q);
        sycl::free(output_d, q);
        sycl::free(neighbors_d, q);
    }

  private:
    std::size_t problem_size_;
    int num_iters_;
    int num_runs_;
    std::size_t local_size_;
    int neigh_count_;
    float cutsq_;
    float lj1_;
    float lj2_;

    void init_data(std::vector<sycl::float4>& input, std::vector<int>& neighbors) const {
        for (std::size_t i = 0; i < problem_size_; ++i) {
            input[i] = sycl::float4(static_cast<float>(i),
                                    static_cast<float>(i % 251),
                                    static_cast<float>(i % 127),
                                    0.0f);
        }

        for (int j = 0; j < neigh_count_; ++j) {
            for (std::size_t i = 0; i < problem_size_; ++i) {
                neighbors[static_cast<std::size_t>(j) * problem_size_ + i] =
                    static_cast<int>((i + static_cast<std::size_t>(j) + 1) % problem_size_);
            }
        }
    }
};

void print_usage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name
              << " <size> <iters> <csv_dir> <freq> [runs] [local_size] [neigh_count]\n"
              << "  " << exe_name
              << " --size=<n> --iters=<n> [--csv-path=<dir>] [--freq=<mhz>] [--runs=<n>] [--local-size=<n>] [--neigh-count=<n>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    MolDynSimulation sim;

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
            if (argc >= 8) {
                sim.SetNeighborCount(std::stoi(argv[7]));
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
                } else if (key == "neigh-count") {
                    sim.SetNeighborCount(std::stoi(value));
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
