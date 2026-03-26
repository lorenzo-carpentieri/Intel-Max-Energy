#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <synergy.hpp>

#include "../include/utils/utils.hpp"

namespace {

constexpr float ZERO = 0.0f;
constexpr float ONE = 1.0f;
constexpr float HALF = 0.5f;
constexpr float A1 = 0.319381530f;
constexpr float A2 = -0.356563782f;
constexpr float A3 = 1.781477937f;
constexpr float A4 = -1.821255978f;
constexpr float A5 = 1.330274429f;
constexpr float INV_ROOT2PI = 0.39894228f;
constexpr float NCDF = 0.2316419f;

constexpr std::size_t DEFAULT_SIZE = 1 << 20;
constexpr int DEFAULT_ITERS = 100;
constexpr int DEFAULT_RUNS = 5;
constexpr std::size_t DEFAULT_LOCAL_SIZE = 256;

std::string csv_path;
synergy::frequency freq = 1600;

std::string build_csv_file_path(const std::string& csv_dir) {
    if (csv_dir.empty()) {
        return {};
    }
    return csv_dir + "/black_scholes_" + std::to_string(freq) + ".csv";
}

class BlackScholesSimulation {
  public:
    BlackScholesSimulation()
        : problem_size_(DEFAULT_SIZE),
          num_iters_(DEFAULT_ITERS),
          num_runs_(DEFAULT_RUNS),
          local_size_(DEFAULT_LOCAL_SIZE) {}

    void SetProblemSize(std::size_t size) { problem_size_ = std::max<std::size_t>(1, size); }
    void SetNumIters(int iters) { num_iters_ = std::max(1, iters); }
    void SetNumRuns(int runs) { num_runs_ = std::max(1, runs); }
    void SetLocalSize(std::size_t local_size) {
        local_size_ = std::max<std::size_t>(1, local_size);
    }

    void Start() {
        std::vector<std::uint32_t> cpflag(problem_size_);
        std::vector<float> s0(problem_size_);
        std::vector<float> strike(problem_size_);
        std::vector<float> rate(problem_size_);
        std::vector<float> sigma(problem_size_);
        std::vector<float> maturity(problem_size_);
        std::vector<float> answer(problem_size_, 0.0f);
        init_data(cpflag, s0, strike, rate, sigma, maturity, answer);

        int dev_id = 0;
        synergy::frequency uncore_freq = 0;
        sycl::device device = utils::device::pick_device(dev_id);
        synergy::queue q(uncore_freq, freq, device);

        synergy::utils::check_core_freq(q.get_synergy_device(), freq, 1000);

        const std::size_t global_size =
            ((problem_size_ + local_size_ - 1) / local_size_) * local_size_;
        const std::size_t problem_size = problem_size_;

        std::uint32_t* cpflag_d = sycl::malloc_device<std::uint32_t>(problem_size, q);
        float* s0_d = sycl::malloc_device<float>(problem_size, q);
        float* strike_d = sycl::malloc_device<float>(problem_size, q);
        float* rate_d = sycl::malloc_device<float>(problem_size, q);
        float* sigma_d = sycl::malloc_device<float>(problem_size, q);
        float* maturity_d = sycl::malloc_device<float>(problem_size, q);
        float* answer_d = sycl::malloc_device<float>(problem_size, q);

        if (cpflag_d == nullptr || s0_d == nullptr || strike_d == nullptr ||
            rate_d == nullptr || sigma_d == nullptr || maturity_d == nullptr ||
            answer_d == nullptr) {
            if (cpflag_d != nullptr) {
                sycl::free(cpflag_d, q);
            }
            if (s0_d != nullptr) {
                sycl::free(s0_d, q);
            }
            if (strike_d != nullptr) {
                sycl::free(strike_d, q);
            }
            if (rate_d != nullptr) {
                sycl::free(rate_d, q);
            }
            if (sigma_d != nullptr) {
                sycl::free(sigma_d, q);
            }
            if (maturity_d != nullptr) {
                sycl::free(maturity_d, q);
            }
            if (answer_d != nullptr) {
                sycl::free(answer_d, q);
            }
            throw std::runtime_error("Device allocation failed for black_scholes");
        }

        q.memcpy(cpflag_d, cpflag.data(), problem_size * sizeof(std::uint32_t)).wait();
        q.memcpy(s0_d, s0.data(), problem_size * sizeof(float)).wait();
        q.memcpy(strike_d, strike.data(), problem_size * sizeof(float)).wait();
        q.memcpy(rate_d, rate.data(), problem_size * sizeof(float)).wait();
        q.memcpy(sigma_d, sigma.data(), problem_size * sizeof(float)).wait();
        q.memcpy(maturity_d, maturity.data(), problem_size * sizeof(float)).wait();
        q.memcpy(answer_d, answer.data(), problem_size * sizeof(float)).wait();

        std::vector<synergy::device> synergy_devs;
        synergy_devs.push_back(q.get_synergy_device());
        synergy::profiler::PowerProfiler power_prof(synergy_devs, 10);

        std::cout << "[INFO] Device: "
                  << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "[INFO] black_scholes problem_size: " << problem_size
                  << ", iterations: " << num_iters_
                  << ", runs: " << num_runs_
                  << ", local_size: " << local_size_ << '\n';

        for (int run_id = 0; run_id < num_runs_; ++run_id) {
            power_prof.start();
            utils::data_types::energy_t start_energy = q.device_energy_consumption();
            auto time_start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters_; ++iter) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<class BlackScholesStandaloneKernel>(
                        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size_)),
                        [=](sycl::nd_item<1> item) {
                            const std::size_t tid = item.get_global_linear_id();
                            if (tid >= problem_size) {
                                return;
                            }

                            const std::uint32_t option_type = cpflag_d[tid];
                            const float s0_value = s0_d[tid];
                            const float strike_value = strike_d[tid];
                            const float rate_value = rate_d[tid];
                            const float sigma_value = sigma_d[tid];
                            const float maturity_value = maturity_d[tid];

                            float d1 = sycl::log(s0_value / strike_value) +
                                       (rate_value + HALF * sigma_value * sigma_value) * maturity_value;
                            d1 = d1 / (sigma_value * sycl::sqrt(maturity_value));
                            const float expval = sycl::exp(ZERO - rate_value * maturity_value);
                            float d2 = d1 - sigma_value * sycl::sqrt(maturity_value);

                            const bool flag1 = d1 < ZERO;
                            const bool flag2 = d2 < ZERO;
                            d1 = sycl::fabs(d1);
                            d2 = sycl::fabs(d2);

                            const float k1 = ONE / (ONE + NCDF * d1);
                            const float k2 = ONE / (ONE + NCDF * d2);

                            float accum1 = A4 + A5 * k1;
                            float accum2 = A4 + A5 * k2;
                            accum1 = k1 * accum1 + A3;
                            accum2 = k2 * accum2 + A3;
                            accum1 = k1 * accum1 + A2;
                            accum2 = k2 * accum2 + A2;
                            accum1 = k1 * accum1 + A1;
                            accum2 = k2 * accum2 + A1;
                            accum1 = k1 * accum1;
                            accum2 = k2 * accum2;

                            float n1 = sycl::exp(ZERO - HALF * d1 * d1);
                            float n2 = sycl::exp(ZERO - HALF * d2 * d2);
                            n1 *= INV_ROOT2PI;
                            n2 *= INV_ROOT2PI;

                            const float candidate_answer1 = ONE - n1 * accum1;
                            const float candidate_answer2 = ONE - n2 * accum2;
                            const float nd1 = flag1 ? (ONE - candidate_answer1) : candidate_answer1;
                            const float nd2 = flag2 ? (ONE - candidate_answer2) : candidate_answer2;

                            const float call = s0_value * nd1 - strike_value * expval * nd2;
                            const float put =
                                strike_value * expval * (ONE - nd2) - s0_value * (ONE - nd1);
                            answer_d[tid] = option_type ? call : put;
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

            q.memcpy(answer.data(), answer_d, problem_size * sizeof(float)).wait();

            const std::size_t sample_idx = problem_size / 2;
            std::cout << "[INFO] Run " << run_id
                      << ": time_ms=" << time_ms
                      << ", device_energy_mj=" << total_dev_energy
                      << ", sample_answer=" << answer[sample_idx] << '\n';

            if (!csv_path.empty()) {
                utils::logger::Logger log(build_csv_file_path(csv_path));
                synergy::power_trace_t power_trace = power_prof.get_power_execution_data()[0];
                synergy::freq_trace_t freq_trace = power_prof.get_freq_execution_data()[0];
                synergy::temperature_trace_t temp_trace =
                    power_prof.get_temperature_execution_data()[0];
                temp_trace.clear();

                utils::logger::Logger::ProfilingInfo<float> prof_info{
                    "h",
                    "black_scholes",
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

        sycl::free(cpflag_d, q);
        sycl::free(s0_d, q);
        sycl::free(strike_d, q);
        sycl::free(rate_d, q);
        sycl::free(sigma_d, q);
        sycl::free(maturity_d, q);
        sycl::free(answer_d, q);
    }

  private:
    std::size_t problem_size_;
    int num_iters_;
    int num_runs_;
    std::size_t local_size_;

    static void init_data(std::vector<std::uint32_t>& cpflag,
                          std::vector<float>& s0,
                          std::vector<float>& strike,
                          std::vector<float>& rate,
                          std::vector<float>& sigma,
                          std::vector<float>& maturity,
                          std::vector<float>& answer) {
        const float s0_array[4] = {42.0f, 30.0f, 54.0f, 66.0f};
        const float strike_array[16] = {
            40.0f, 36.0f, 44.0f, 48.0f, 24.0f, 28.0f, 32.0f, 36.0f,
            48.0f, 52.0f, 56.0f, 60.0f, 60.0f, 64.0f, 68.0f, 72.0f};
        const float rate_array[4] = {0.1f, 0.09f, 0.11f, 0.12f};
        const float sigma_array[4] = {0.2f, 0.15f, 0.25f, 0.30f};
        const float maturity_array[4] = {0.5f, 0.25f, 0.75f, 1.0f};

        int idx = 0;
        for (std::size_t k = 0; k < cpflag.size(); ++k, ++idx) {
            const int tdex = (idx >> 1) & 0x3;
            const int sigdex = (idx >> 3) & 0x3;
            const int rdex = (idx >> 5) & 0x3;
            const int s0kdex = (idx >> 7) & 0xf;

            cpflag[k] = (idx & 1) ? 0xffffffffu : 0u;
            s0[k] = s0_array[s0kdex >> 2];
            strike[k] = strike_array[s0kdex];
            rate[k] = rate_array[rdex];
            sigma[k] = sigma_array[sigdex];
            maturity[k] = maturity_array[tdex];
            answer[k] = 0.0f;
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
    BlackScholesSimulation sim;

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
