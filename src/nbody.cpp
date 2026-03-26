//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <synergy.hpp>
#include "../include/utils/utils.hpp"
using RealType = float;
constexpr int DEFAULT_RUNS = 5;

struct Particle {
 public:
  Particle() : pos{}, vel{}, acc{}, mass{} {};
  RealType pos[3];
  RealType vel[3];
  RealType acc[3];
  RealType mass;
};

void accelerate_particles_ref( Particle* p, const int n, const float kSofteningSquared, const float kG )
{
  for (int i = 0; i < n; i++) {
    auto pi = p[i];
    RealType acc0 = pi.acc[0];
    RealType acc1 = pi.acc[1];
    RealType acc2 = pi.acc[2];
    for (int j = 0; j < n; j++) {
      RealType dx, dy, dz;
      RealType distance_sqr = 0.0f;
      RealType distance_inv = 0.0f;

      auto pj = p[j];
      dx = pj.pos[0] - pi.pos[0];  // 1flop
      dy = pj.pos[1] - pi.pos[1];  // 1flop
      dz = pj.pos[2] - pi.pos[2];  // 1flop

      distance_sqr =
        dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
      distance_inv = 1.0f / std::sqrt(distance_sqr);       // 1div+1sqrt

      acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
      acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
    }
    pi.acc[0] = acc0;
    pi.acc[1] = acc1;
    pi.acc[2] = acc2;
    p[i] = pi;
  }
}

void update_particles_ref(Particle *__restrict__ p, RealType *__restrict__ e, const int n, RealType dt)
{
  for (int i = 0; i < n; i++) {
    auto pi = p[i];

    pi.vel[0] += pi.acc[0] * dt;  // 2flops
    pi.vel[1] += pi.acc[1] * dt;  // 2flops
    pi.vel[2] += pi.acc[2] * dt;  // 2flops

    pi.pos[0] += pi.vel[0] * dt;  // 2flops
    pi.pos[1] += pi.vel[1] * dt;  // 2flops
    pi.pos[2] += pi.vel[2] * dt;  // 2flops

    pi.acc[0] = 0.f;
    pi.acc[1] = 0.f;
    pi.acc[2] = 0.f;

    e[i] = pi.mass *
      (pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
       pi.vel[2] * pi.vel[2]);  // 7flops

    p[i] = pi;
  }
}

void
accumulate_energy_ref(RealType *e, const int n)
{
  for (int i = 1; i < n; i++) e[0] += e[i];
}



std::string csv_path;
synergy::frequency freq = 1600;

std::string build_csv_file_path(const std::string& csv_dir) {
  if (csv_dir.empty()) {
    return {};
  }
  return csv_dir + "/nbody_" + std::to_string(freq) + ".csv";
}


class GSimulation {
 public:
  GSimulation();

  void Init();
  void SetNumberOfParticles(int N);
  void SetNumberOfSteps(int N);
  void SetNumRuns(int N);
  void Start();
  void Verify();

 private:
  //  Particle *particles_;
  std::vector<Particle> particles_;
  int npart_;       // number of particles
  int nsteps_;      // number of integration steps
  int num_runs_;    // number of benchmark runs
  RealType tstep_;  // time step of the simulation

  int sfreq_;  // sample frequency

  RealType kenergy_;  // kinetic energy
  RealType ref_kenergy_;  // kinetic energy

  double total_time_;   // total time of the simulation
  double total_flops_;  // total number of FLOPS

  void InitPos();
  void InitVel();
  void InitAcc();
  void InitMass();

  void set_npart(const int &N) { npart_ = N; }
  int get_npart() const { return npart_; }

  void set_tstep(const RealType &dt) { tstep_ = dt; }
  RealType get_tstep() const { return tstep_; }

  void set_nsteps(const int &n) { nsteps_ = n; }
  int get_nsteps() const { return nsteps_; }

  void set_num_runs(const int &n) { num_runs_ = n; }
  int get_num_runs() const { return num_runs_; }

  void set_sfreq(const int &sf) { sfreq_ = sf; }
  int get_sfreq() const { return sfreq_; }

  void PrintHeader();
};

void GSimulation::Verify() {
  RealType dt = get_tstep();
  int n = get_npart();
  std::vector<RealType> energy(n, 0.f);
  // allocate particles
  particles_.resize(n);

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

#ifdef DEBUG
  PrintHeader();
#endif

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  Particle *p = particles_.data();;

  RealType *e = energy.data();

  int nsteps = get_nsteps();
  // Looping across integration steps
  for (int s = 1; s <= nsteps; ++s) {

    accelerate_particles_ref(p, n, kSofteningSquared, kG);
    update_particles_ref(p, e, n, dt);
    accumulate_energy_ref(e, n);

    ref_kenergy_ = 0.5 * energy[0];
  }  // end of the time step loop
  std::cout << "\n";
  bool ok = fabsf(kenergy_ - ref_kenergy_) < 1e-3f;
  printf("%s\n", ok ? "PASS" : "FAIL");
}


// The TimeInterval is a simple RAII class.
// Construct the timer at the point you want to start timing.
// Use the Elapsed() method to return time since construction.

class TimeInterval {
 public:
  TimeInterval() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

 private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};



/* Default Constructor for the GSimulation class which sets up the default
 * values for number of particles, number of integration steps, time steo and
 * sample frequency */
GSimulation::GSimulation() {
  std::cout << "==============================="
            << "\n";
  std::cout << " Initialize Gravity Simulation"
            << "\n";
  set_npart(16000);
  set_nsteps(10);
  set_num_runs(DEFAULT_RUNS);
  set_tstep(0.1);
  set_sfreq(1);
}

/* Set the number of particles */
void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }

/* Set the number of integration steps */
void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }

/* Set the number of benchmark runs */
void GSimulation::SetNumRuns(int N) { set_num_runs(N); }

/* Initialize the position of all the particles using random number generator
 * between 0 and 1.0 */
void GSimulation::InitPos() {
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].pos[0] = unif_d(gen);
    particles_[i].pos[1] = unif_d(gen);
    particles_[i].pos[2] = unif_d(gen);
  }
}

/* Initialize the velocity of all the particles using random number generator
 * between -1.0 and 1.0 */
void GSimulation::InitVel() {
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[2] = unif_d(gen) * 1.0e-3f;
  }
}

/* Initialize the acceleration of all the particles to 0 */
void GSimulation::InitAcc() {
  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].acc[0] = 0.f;
    particles_[i].acc[1] = 0.f;
    particles_[i].acc[2] = 0.f;
  }
}

/* Initialize the mass of all the particles using a random number generator
 * between 0 and 1 */
void GSimulation::InitMass() {
  RealType n = static_cast<RealType>(get_npart());
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles_[i].mass = n * unif_d(gen);
  }
}

/* This function does the simulation logic for Nbody */
void GSimulation::Start() {
  RealType dt = get_tstep();
  int n = get_npart();
  std::vector<RealType> energy(n, 0.f);
  // allocate particles
  particles_.resize(n);

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

#ifdef DEBUG
  PrintHeader();
#endif

  total_time_ = 0.;

  constexpr float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  constexpr float kG = 6.67259e-11f;
  double gflops = 1e-9 * ((11. + 18.) * n * n + n * 19.);
  int nf = 0;
  double av = 0.0, dev = 0.0;

  int dev_id=0; // Select the first tile
  synergy::frequency uncore_freq = 0; // not used in this application but can be set from command line arguments for profiling purposes --- IGNORE ---
  sycl::device sycl_device = utils::device::pick_device(dev_id); 
  // Create a queue to the selected device and enabled asynchronous exception
  // handling for that queue
  synergy::queue q(uncore_freq, freq, sycl_device);


  /*
    Set frequency for the GPU device
  */
  synergy::utils::check_core_freq(q.get_synergy_device(), freq, 1000);


  sycl::range<1> gws ((n+255)/256 * 256);
  sycl::range<1> lws (256);

  Particle *p = sycl::malloc_device<Particle>(n, q);
  q.memcpy(p, particles_.data(), n * sizeof(Particle)).wait();

  RealType *e = sycl::malloc_device<RealType>(n, q);

  TimeInterval t0;
  int nsteps = get_nsteps();
  int num_runs = get_num_runs();
  // Looping across integration steps
  std::vector<synergy::device> synergy_devs;
	synergy_devs.push_back(q.get_synergy_device());
  synergy::profiler::PowerProfiler power_prof(synergy_devs, 10);
	for(int run_id = 0; run_id < num_runs; run_id++) {
		std::cout << "Run id: " << run_id << std::endl;
		power_prof.start();
		utils::data_types::energy_t start_energy = q.device_energy_consumption();
		auto time_start = std::chrono::high_resolution_clock::now();
		for (int s = 1; s <= nsteps; ++s) {
			TimeInterval ts0;

			// Submitting first kernel to device which computes acceleration of all
			// particles
			q.submit([&](sycl::handler& h) {
				h.parallel_for<class compute_acceleration>(
					sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
					int i = item.get_global_id(0);
					if (i >= n) return;

					auto pi = p[i];
					RealType acc0 = pi.acc[0];
					RealType acc1 = pi.acc[1];
					RealType acc2 = pi.acc[2];
					for (int j = 0; j < n; j++) {
						RealType dx, dy, dz;
						RealType distance_sqr = 0.0f;
						RealType distance_inv = 0.0f;

						auto pj = p[j];
						dx = pj.pos[0] - pi.pos[0];  // 1flop
						dy = pj.pos[1] - pi.pos[1];  // 1flop
						dz = pj.pos[2] - pi.pos[2];  // 1flop

						distance_sqr =
								dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
						distance_inv = sycl::rsqrt(distance_sqr);       // 1div+1sqrt

						acc0 += dx * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
						acc1 += dy * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
						acc2 += dz * kG * pj.mass * distance_inv * distance_inv * distance_inv;  // 6flops
					}
					pi.acc[0] = acc0;
					pi.acc[1] = acc1;
					pi.acc[2] = acc2;
					p[i] = pi;
				});
			});
			// Second kernel updates the velocity and position for all particles
			q.submit([&](sycl::handler& h) {
				h.parallel_for<class update_velocity_position>(
					sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
					auto i = item.get_global_id(0);
					if (i >= n) return;

					auto pi = p[i];

					pi.vel[0] += pi.acc[0] * dt;  // 2flops
					pi.vel[1] += pi.acc[1] * dt;  // 2flops
					pi.vel[2] += pi.acc[2] * dt;  // 2flops

					pi.pos[0] += pi.vel[0] * dt;  // 2flops
					pi.pos[1] += pi.vel[1] * dt;  // 2flops
					pi.pos[2] += pi.vel[2] * dt;  // 2flops

					pi.acc[0] = 0.f;
					pi.acc[1] = 0.f;
					pi.acc[2] = 0.f;

					e[i] = pi.mass *
								(pi.vel[0] * pi.vel[0] + pi.vel[1] * pi.vel[1] +
									pi.vel[2] * pi.vel[2]);  // 7flops

					p[i] = pi;
				});
			});
			/* Third kernel accumulates the energy of this Nbody system
			* Reduction operation can be done using reducer interface in SYCL 2020
			*/
			q.submit([&](sycl::handler& h) {
				h.single_task<class accumulate_energy>([=]() {
					for (int i = 1; i < n; i++) e[0] += e[i];
				});
			});

			q.wait();
			double elapsed_seconds = ts0.Elapsed();

			q.memcpy(energy.data(), e, sizeof(RealType)).wait();

			kenergy_ = 0.5 * energy[0];
			if ((s % get_sfreq()) == 0) {
				nf += 1;
	#ifdef DEBUG
				std::cout << " " << std::left << std::setw(8) << s << std::left
									<< std::setprecision(5) << std::setw(8) << s * get_tstep()
									<< std::left << std::setprecision(5) << std::setw(12)
									<< kenergy_ << std::left << std::setprecision(5)
									<< std::setw(12) << elapsed_seconds << std::left
									<< std::setprecision(5) << std::setw(12)
									<< gflops * get_sfreq() / elapsed_seconds << "\n";
	#endif
				if (nf > 2) {
					av += gflops * get_sfreq() / elapsed_seconds;
					dev += gflops * get_sfreq() * gflops * get_sfreq() /
								(elapsed_seconds * elapsed_seconds);
				}
			}
		}  // end of the time step loop

		power_prof.stop();
    std::cout<< "Run id: " << run_id << " completed" << std::endl;
		auto time_end = std::chrono::high_resolution_clock::now();

		/*********************** Profiling energy info ***********************/
		utils::data_types::energy_t end_energy = q.device_energy_consumption();
		utils::data_types::energy_t total_dev_energy = end_energy - start_energy;
		// Timing
		double time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
		synergy::power_trace_t power_trace = power_prof.get_power_execution_data()[0];
		synergy::freq_trace_t freq_trace = power_prof.get_freq_execution_data()[0];
		synergy::temperature_trace_t temp_trace = power_prof.get_temperature_execution_data()[0];
		temp_trace.clear(); // Temperature data is not available for GPU devices, so we clear the vector to avoid logging invalid data
		utils::logger::Logger::ProfilingInfo<RealType> prof_info{"h","nbody", static_cast<size_t>(n), nsteps, num_runs, run_id, time_ms, total_dev_energy, 0, power_trace  , freq_trace, temp_trace, utils::data_types::GPUMode::FLAT};
		/*********************************************************************/
    if (!csv_path.empty()) {
      utils::logger::Logger log(build_csv_file_path(csv_path));
      log.log_result(prof_info);
    }
		power_prof.clean();
	}
  std::cout<< "End loop" <<  std::endl;
	total_time_ = t0.Elapsed();
  total_flops_ = gflops * get_nsteps();
  av /= (double)(nf - 2);
  dev = std::sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Energy        : " << kenergy_ << "\n";
  std::cout << "# Total Time (s)      : " << total_time_ << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "===============================";
  std::cout << "\n";






  sycl::free(p, q);
  sycl::free(e, q);
  return;
}

#ifdef DEBUG
/* Print the headers for the output */
void GSimulation::PrintHeader() {
  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << "\n";

  std::cout << "------------------------------------------------"
            << "\n";
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
            << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
            << std::left << std::setw(12) << "time (s)" << std::left
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << "------------------------------------------------"
            << "\n";
}
#endif

void print_usage(const char* exe_name) {
  std::cout << "Usage:\n"
            << "  " << exe_name << " <size> <iters> <csv_dir> <freq> [runs]\n"
            << "  " << exe_name
            << " --size=<n> --iters=<n> [--csv-path=<dir>] [--freq=<mhz>] [--runs=<n>]\n";
}


int main(int argc, char** argv) {
  GSimulation sim;

  try {
    if (argc > 1 && std::string(argv[1]) == "--help") {
      print_usage(argv[0]);
      return 0;
    }

    if (argc > 1 && std::string(argv[1]).rfind("--", 0) != 0) {
      sim.SetNumberOfParticles(std::stoi(argv[1]));
      if (argc >= 3) {
        sim.SetNumberOfSteps(std::stoi(argv[2]));
      }
      if (argc >= 4) {
        csv_path = std::string(argv[3]);
      }
      if (argc >= 5) {
        freq = static_cast<synergy::frequency>(std::stoul(argv[4]));
      }
      if (argc >= 6) {
        sim.SetNumRuns(std::stoi(argv[5]));
      }
    } else {
      utils::arg_parse::map_t params = utils::arg_parse::parse(argc, argv);
      for (const auto& [key, value] : params) {
        if (key == "size") {
          sim.SetNumberOfParticles(std::stoi(value));
        } else if (key == "iters") {
          sim.SetNumberOfSteps(std::stoi(value));
        } else if (key == "csv-path") {
          csv_path = value;
        } else if (key == "freq") {
          freq = static_cast<synergy::frequency>(std::stoul(value));
        } else if (key == "runs") {
          sim.SetNumRuns(std::stoi(value));
        } else {
          throw std::runtime_error("Unknown parameter: " + key);
        }
      }
    }

    sim.Start();
    // sim.Verify();
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }

  return 0;
}
