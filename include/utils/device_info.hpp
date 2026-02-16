#pragma once

#include <vector>
#include <sycl/sycl.hpp>
#include <iostream>

namespace utils {
    namespace device{
        
        
        size_t max_alloc_size(sycl::device dev){
            return dev.get_info<sycl::info::device::max_mem_alloc_size>();
        }
        
        sycl::device pick_device(int dev_id){
            std::vector<sycl::device> candidates; // Available devices for each rank
            
            for (const auto& plat : sycl::platform::get_platforms()) { // Handle only Level-Zero platforms: oneCCL only supports Intel GPUs 
                auto name = plat.get_info<sycl::info::platform::name>();
                if (name.find("Level-Zero") == std::string::npos) continue;

                for (const auto& root : plat.get_devices()) {
                    if (!root.is_gpu()) continue;
                    candidates.push_back(root);
                }
            }     
            return candidates[dev_id];
        }
        
        
        std::vector<sycl::device> pick_sub_devices(sycl::device dev) {
            auto max_sub_devices = dev.get_info<sycl::info::device::partition_max_sub_devices>();
            std::clog << "[LOG] Max. num. of sub devices: " << max_sub_devices <<std::endl;
            auto tiles = dev.create_sub_devices<
                            sycl::info::partition_property::partition_by_affinity_domain>(
                            sycl::info::partition_affinity_domain::next_partitionable);
                                
            //  auto tiles = dev.create_sub_devices(max_sub_devices);
             return tiles;
        }
        
        size_t compute_num_el_per_tile(const sycl::device& dev) {
            // Max number of parallel compute units (SMs / EUs / CUs)
            size_t max_compute_units =
                dev.get_info<sycl::info::device::max_compute_units>();

            // Max size of a work-group
            size_t max_work_group_size =
                dev.get_info<sycl::info::device::max_work_group_size>();

            constexpr size_t FACTOR = 8;

            return max_compute_units * max_work_group_size * FACTOR;    
        }
        
    }
}