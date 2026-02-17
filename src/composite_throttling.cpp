#include "synergy.hpp"
#include <sycl/sycl.hpp>
#include "../include/kernels.hpp"
#include "../include/utils/utils.hpp"



#define T float
#define MAX_EXEC_TIME_MS 6000

sycl::event submit_kernel(
    sycl::queue& q,
    const std::string& kernel_type,
    T* data,
    size_t n,
    int iters
) {
    if (kernel_type == "comp") {
        return q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(n),
                kernels::ComputeKernel<T>(data, n, iters)
            );
        });
    }
    else if (kernel_type == "mem") {
        return q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(n),
                kernels::MemoryBoundKernel<T>(data, n, iters)
            );
        });
    }
    else if(kernel_type == "none"){
        return sycl::event();
    }
    else {
        throw std::runtime_error("Unknown kernel type: " + kernel_type);
    }
}


int main(int argc, char* argv[]){
    std::string csv_path;
    int N_ITERS=1;
    int NUM_RUNS=1; 
    synergy::frequency freq0=1600;
    synergy::frequency freq1=1600;
    std::string kernel1;
    std::string kernel2;
    
    // TODO: handle with arg_parse
    utils::arg_parse::map_t params = utils::arg_parse::parse(argc, argv);
    for (const auto& [key, value] : params) {
        const std::string& s = value;

        if (key == "csv-path") {
            csv_path = s + "/composite_throttling_";
        }
        else if (key == "runs") {
            NUM_RUNS = std::stoi(s);
        }
        else if (key == "freq0") {
            freq0 = static_cast<synergy::frequency>(std::stoul(s));
        }
        else if (key == "freq1") {
            freq1 = static_cast<synergy::frequency>(std::stoul(s));
        }
        else if(key=="kernel1"){
            kernel1=s;
        }
        else if(key=="kernel2"){
            kernel2=s;
        }
        else {
            throw std::runtime_error("Unknown parameter: " + key);
        }
        
        // std::cout << std::endl;
    }
    csv_path += "f" + std::to_string(freq0) + "_" + "f" + std::to_string(freq1) + "_k1" + kernel1 + "_k2" + kernel2 + ".csv";
    // std::visit([](auto&& arg) { std::cout << arg; }, value);
    std::cout << csv_path << std::endl;
    

    utils::logger::Logger log(csv_path);

    synergy::frequency core_freq_tile0=freq0;
    synergy::frequency core_freq_tile1=freq1;
    
    synergy::frequency uncore_freq=0; // In SYnergy 0 corrsponds to default freq.


    int dev_id=0;
    sycl::device dev = utils::device::pick_device(dev_id); 
    std::vector<sycl::device> tiles = utils::device::pick_sub_devices(dev);

    
    // SYnergy queue for each sub device used to change the frequency
    synergy::queue q_tile0(uncore_freq, core_freq_tile0, tiles[0]);
    synergy::queue q_tile1(uncore_freq, core_freq_tile1, tiles[1]);
    synergy::queue q_total(dev); // Creaate the synergy queue for the enitre GPU.

    // SYCL queue to run the kernel i concurency. 
    // SYnergy queu automatically serialize kernel execution.
    sycl::queue q0(tiles[0], sycl::property_list{sycl::property::queue::enable_profiling{}});
    sycl::queue q1(tiles[1], sycl::property_list{sycl::property::queue::enable_profiling{}});

    std::vector<synergy::device> synergy_devs;
    synergy_devs.push_back(q_tile0.get_synergy_device());
    synergy_devs.push_back(q_tile1.get_synergy_device());
    synergy_devs.push_back(q_total.get_synergy_device());


   
    /*******************  Polling frequecny **************/
    synergy::utils::check_core_freq(q_tile0.get_synergy_device(), core_freq_tile0, 1000);
    synergy::utils::check_core_freq(q_tile1.get_synergy_device(), core_freq_tile1, 1100);
    
    // Check core frequency
    std::cout << "[INFO]: Tile 0 core freq: "<< q_tile0.get_synergy_device().get_core_frequency(false) <<std::endl;
    std::cout << "[INFO]: Tile 1 core freq: "<< q_tile1.get_synergy_device().get_core_frequency(false) <<std::endl;
    /****************** End polling freq *******************/
    
    /*************  Start allocate data *****************/
    std::cout << "[INFO]: Start data allocation ... " << std::endl;

    size_t num_el_per_tile = std::numeric_limits<int>::max();
    std::cout << "[INFO]: Num elements per tile: " << num_el_per_tile << std::endl;
    
    // Host data allocation
    T* data_tile0_h = (T*) malloc(num_el_per_tile * sizeof(T));
    T* data_tile1_h = (T*) malloc(sizeof(T) * num_el_per_tile);

    // Allocate data on the subdevices
    T* data_tile0_d = (T*) sycl::malloc_device(sizeof(T) * num_el_per_tile, q_tile0);
    T* data_tile1_d = (T*) sycl::malloc_device(sizeof(T) * num_el_per_tile, q_tile1);

    // Init data on the host
    for(int i = 0; i < num_el_per_tile; i++){
        data_tile0_h[i] = static_cast<T>(i);
        data_tile1_h[i] = static_cast<T>(i);
    }

    // Copy from host to device
    q_tile0.memcpy(data_tile0_d, data_tile0_h, sizeof(T) * num_el_per_tile);
    q_tile1.memcpy(data_tile1_d, data_tile1_h, sizeof(T) * num_el_per_tile);
    
    std::cout << "[INFO]: End data allocation" << std::endl;

    /************** End data allocation **************/
  

    /************** Start compute Num iteres  *************/
    std::cout << "[INFO]: Calculate num iters" << std::endl;
    double time_ms_tile0_kern = 0;
    int N_ITERS_0 = 1;

    while (time_ms_tile0_kern < MAX_EXEC_TIME_MS){

        sycl::event e0_tile0 = submit_kernel(q0, kernel1, data_tile0_d, num_el_per_tile, N_ITERS_0);
        
        e0_tile0.wait();

    
        time_ms_tile0_kern = utils::profiling::get_kernel_time_ms(e0_tile0);
        int increment_iters= MAX_EXEC_TIME_MS / time_ms_tile0_kern;
        N_ITERS_0 += 1000;
        std::cout << "[INFO]: Kernel time: " << time_ms_tile0_kern << std::endl;
    }

    double time_ms_tile1_kern = 0;
    int N_ITERS_1 = 1;
    sycl::event e0_tile1;

    if (kernel2 != "none"){
        while (time_ms_tile1_kern < MAX_EXEC_TIME_MS){
            e0_tile1 = submit_kernel(q1, kernel2, data_tile1_d, num_el_per_tile, N_ITERS_1);
            e0_tile1.wait();
            time_ms_tile1_kern = utils::profiling::get_kernel_time_ms(e0_tile1);
            int increment_iters = MAX_EXEC_TIME_MS / time_ms_tile1_kern;
            N_ITERS_1 += 1000;
            std::cout << "[INFO]: Kernel time: " << time_ms_tile1_kern << std::endl;
            N_ITERS_1 += increment_iters;
        }
    }
         
    /************** End compute Num iteres  *************/

    // Create power profiler
    synergy::profiler::PowerProfiler power_prof(synergy_devs, 1);

    /************** Start run benchmark *******************/
    std::cout << "[INFO]: Start benchmark run" << std::endl;

    for (int run_id = 0; run_id < NUM_RUNS; run_id++){
        // Start power profiling for both tiles
        power_prof.start();
        /********** Start running kernels ***************/
        sycl::event e0_tile0 = submit_kernel(q0, kernel1, data_tile0_d, num_el_per_tile, N_ITERS_0);

        if (kernel2 != "none")
            e0_tile1 = submit_kernel(q1, kernel2, data_tile1_d, num_el_per_tile, N_ITERS_1);
  

        e0_tile0.wait();

        if (kernel2 != "none")
            e0_tile1.wait();
        /********** End running kernels ***************/
        power_prof.stop();

        /********** Start profiling ******************/
        utils::data_types::energy_t tile0_gpu_energy = q_tile0.device_energy_consumption();
        utils::data_types::energy_t tile1_gpu_energy = q_tile1.device_energy_consumption();

        // Timing
        double time_ms_tile0_kern = utils::profiling::get_kernel_time_ms(e0_tile0);
        
        if (kernel2=="none") // no kernel on the second tile
            time_ms_tile1_kern = 0;
        else
            time_ms_tile1_kern = utils::profiling::get_kernel_time_ms(e0_tile1);
        std::vector<synergy::power_trace_t> power_traces = power_prof.get_power_execution_data();
        std::vector<synergy::freq_trace_t> freq_traces = power_prof.get_freq_execution_data();
        std::vector<synergy::temperature_trace_t> temp_traces = power_prof.get_temperature_execution_data();

        synergy::power_trace_t power_trace_tile0 = power_traces[0];
        synergy::power_trace_t power_trace_tile1 = power_traces[1];
        synergy::power_trace_t power_trace_total = power_traces[2];


        synergy::freq_trace_t freq_trace_tile0 = freq_traces[0];
        synergy::freq_trace_t freq_trace_tile1 = freq_traces[1];
        synergy::freq_trace_t freq_trace_total = freq_traces[2];

        synergy::power_trace_t temp_trace_tile0 = temp_traces[0];
        synergy::power_trace_t temp_trace_tile1 = temp_traces[1];
        synergy::power_trace_t temp_trace_total = temp_traces[2];

        utils::logger::Logger::ProfilingInfo<T> prof_info_tile0{"h","composite_throttling_tile0", num_el_per_tile, N_ITERS_0, NUM_RUNS, run_id, time_ms_tile0_kern, tile0_gpu_energy, 0, power_trace_tile0  , freq_trace_tile0, temp_trace_tile0, utils::data_types::GPUMode::COMPOSITE};
        utils::logger::Logger::ProfilingInfo<T> prof_info_tile1{"h","composite_throttling_tile1", num_el_per_tile, N_ITERS_1, NUM_RUNS, run_id, time_ms_tile1_kern, tile1_gpu_energy, 0, power_trace_tile1 , freq_trace_tile1, temp_trace_tile1, utils::data_types::GPUMode::COMPOSITE};
        utils::logger::Logger::ProfilingInfo<T> prof_info_gpu{"h","composite_throttling_total", num_el_per_tile, N_ITERS_1, NUM_RUNS, run_id, time_ms_tile1_kern, tile1_gpu_energy, 0, power_trace_total , freq_trace_total, temp_trace_total, utils::data_types::GPUMode::COMPOSITE};
        
        log.log_result(prof_info_tile0);
        log.log_result(prof_info_tile1);
        log.log_result(prof_info_gpu);

        power_prof.clean();


        /********** End profiling ******************/
    }
    
    /************** End run benchmark *******************/

    
    // Copy device to host 
    q_tile0.memcpy(data_tile0_h, data_tile0_d, sizeof(T) * num_el_per_tile);
    q_tile1.memcpy(data_tile1_h, data_tile1_d, sizeof(T) * num_el_per_tile);


    // Check results
    for(int i = 0; i < 10; i++)
        std::cout << data_tile0_h[i] << ", " << data_tile1_h[i] << std::endl;


    // Free memory
    sycl::free(data_tile0_d, q_tile0);
    sycl::free(data_tile1_d, q_tile1);

    free(data_tile0_h);
    free(data_tile1_h);


    return 0;
}