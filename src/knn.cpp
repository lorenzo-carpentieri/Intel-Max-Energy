#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
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
constexpr std::size_t DEFAULT_NUM_REF = 100000;

std::string csv_path;
synergy::frequency freq = 1600;

std::string build_csv_file_path(const std::string& csv_dir) {
    if (csv_dir.empty()) {
        return {};
    }
    return csv_dir + "/knn_" + std::to_string(freq) + ".csv";
}

class KnnSimulation {
  public:
    KnnSimulation()
        : problem_size_(DEFAULT_SIZE),
          num_iters_(DEFAULT_ITERS),
          num_runs_(DEFAULT_RUNS),
          local_size_(DEFAULT_LOCAL_SIZE),
          num_ref_(DEFAULT_NUM_REF) {}

    void SetProblemSize(std::size_t size) { problem_size_ = std::max<std::size_t>(1, size); }
    void SetNumIters(int iters) { num_iters_ = std::max(1, iters); }
    void SetNumRuns(int runs) { num_runs_ = std::max(1, runs); }
    void SetLocalSize(std::size_t local_size) {
        local_size_ = std::max<std::size_t>(1, local_size);
    }
    void SetNumRef(std::size_t num_ref) { num_ref_ = std::max<std::size_t>(1, num_ref); }

    void Start() {
        std::vector<float> ref(num_ref_);
        std::vector<float> query(problem_size_);
        std::vector<float> dists(problem_size_, 0.0f);
        std::vector<int> neighbors(problem_size_, -1);
        init_data(ref, query);

        int dev_id = 0;
        synergy::frequency uncore_freq = 0;
        sycl::device device = utils::device::pick_device(dev_id);
        synergy::queue q(uncore_freq, freq, device);

        synergy::utils::check_core_freq(q.get_synergy_device(), freq, 1000);

        const std::size_t global_size =
            ((problem_size_ + local_size_ - 1) / local_size_) * local_size_;
        const std::size_t problem_size = problem_size_;
        const std::size_t num_ref = num_ref_;

        float* ref_d = sycl::malloc_device<float>(num_ref, q);
        float* query_d = sycl::malloc_device<float>(problem_size, q);
        float* dists_d = sycl::malloc_device<float>(problem_size, q);
        int* neighbors_d = sycl::malloc_device<int>(problem_size, q);

        if (ref_d == nullptr || query_d == nullptr || dists_d == nullptr || neighbors_d == nullptr) {
            if (ref_d != nullptr) {
                sycl::free(ref_d, q);
            }
            if (query_d != nullptr) {
                sycl::free(query_d, q);
            }
            if (dists_d != nullptr) {
                sycl::free(dists_d, q);
            }
            if (neighbors_d != nullptr) {
                sycl::free(neighbors_d, q);
            }
            throw std::runtime_error("Device allocation failed for KNN");
        }

        q.memcpy(ref_d, ref.data(), num_ref * sizeof(float)).wait();
        q.memcpy(query_d, query.data(), problem_size * sizeof(float)).wait();
        q.memcpy(dists_d, dists.data(), problem_size * sizeof(float)).wait();
        q.memcpy(neighbors_d, neighbors.data(), problem_size * sizeof(int)).wait();

        std::vector<synergy::device> synergy_devs;
        synergy_devs.push_back(q.get_synergy_device());
        synergy::profiler::PowerProfiler power_prof(synergy_devs, 10);

        std::cout << "[INFO] Device: "
                  << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "[INFO] KNN problem_size: " << problem_size
                  << ", num_ref: " << num_ref
                  << ", iterations: " << num_iters_
                  << ", runs: " << num_runs_
                  << ", local_size: " << local_size_ << '\n';

        for (int run_id = 0; run_id < num_runs_; ++run_id) {
            power_prof.start();
            utils::data_types::energy_t start_energy = q.device_energy_consumption();
            auto time_start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters_; ++iter) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<class KnnStandaloneKernel>(
                        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size_)),
                        [=](sycl::nd_item<1> item) {
                            const std::size_t gid = item.get_global_linear_id();
                            if (gid >= problem_size) {
                                return;
                            }

                            const float query_value = query_d[gid];
                            float cur_dist = std::numeric_limits<float>::max();
                            int cur_neighbor = 0;

                            for (std::size_t i = 0; i < num_ref; ++i) {
                                const float diff = ref_d[i] - query_value;
                                const float dist = diff * diff;
                                if (dist < cur_dist) {
                                    cur_dist = dist;
                                    cur_neighbor = static_cast<int>(i);
                                }
                            }

                            dists_d[gid] = sycl::sqrt(cur_dist);
                            neighbors_d[gid] = cur_neighbor;
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

            q.memcpy(dists.data(), dists_d, problem_size * sizeof(float)).wait();
            q.memcpy(neighbors.data(), neighbors_d, problem_size * sizeof(int)).wait();

            const std::size_t sample_idx = problem_size / 2;
            std::cout << "[INFO] Run " << run_id
                      << ": time_ms=" << time_ms
                      << ", device_energy_mj=" << total_dev_energy
                      << ", sample_query=" << query[sample_idx]
                      << ", sample_dist=" << dists[sample_idx]
                      << ", sample_neighbor=" << neighbors[sample_idx] << '\n';

            if (!csv_path.empty()) {
                utils::logger::Logger log(build_csv_file_path(csv_path));
                synergy::power_trace_t power_trace = power_prof.get_power_execution_data()[0];
                synergy::freq_trace_t freq_trace = power_prof.get_freq_execution_data()[0];
                synergy::temperature_trace_t temp_trace =
                    power_prof.get_temperature_execution_data()[0];
                temp_trace.clear();

                utils::logger::Logger::ProfilingInfo<float> prof_info{
                    "h",
                    "knn",
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

        sycl::free(ref_d, q);
        sycl::free(query_d, q);
        sycl::free(dists_d, q);
        sycl::free(neighbors_d, q);
    }

  private:
    std::size_t problem_size_;
    int num_iters_;
    int num_runs_;
    std::size_t local_size_;
    std::size_t num_ref_;

    static void init_data(std::vector<float>& ref, std::vector<float>& query) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (float& value : ref) {
            value = dist(gen);
        }
        for (float& value : query) {
            value = dist(gen);
        }
    }
};

void print_usage(const char* exe_name) {
    std::cout << "Usage:\n"
              << "  " << exe_name << " <size> <iters> <csv_dir> <freq> [runs] [local_size] [nref]\n"
              << "  " << exe_name
              << " --size=<n> --iters=<n> [--csv-path=<dir>] [--freq=<mhz>] [--runs=<n>] [--local-size=<n>] [--nref=<n>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    KnnSimulation sim;

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
                sim.SetNumRef(static_cast<std::size_t>(std::stoull(argv[7])));
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
                } else if (key == "nref") {
                    sim.SetNumRef(static_cast<std::size_t>(std::stoull(value)));
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
