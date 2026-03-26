#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <typeinfo>
#include "data_types.hpp"

namespace utils{
    namespace logger{
        namespace fs = std::filesystem;
        // Utility struct to represent a single field in the CSV log
        // Used to build dynamic CSV rows with extra fields if needed
        struct CsvField {
            std::string key;
            std::string value;

            template<typename T>
            CsvField(std::string k, T&& v)
                    : key(std::move(k))
                    , value(to_string(std::forward<T>(v))) {}

                private:
                    static std::string to_string(const std::string& v) { return v; }
                    static std::string to_string(const char* v) { return v; }

                    template<typename T>
                    static std::string to_string(T v) {
                        std::ostringstream oss;
                        oss << v;
                        return oss.str();
                    }
                };

        class Logger {
        private:
        
            std::string output_file_path;
            std::string bench_name;
            utils::data_types::GPUMode gpu_mode = utils::data_types::GPUMode::COMPOSITE;
            
            int run_id;


            
            void ensure_directory_exists() const {
                fs::path path(output_file_path);
                // Get the parent directory of the file
                fs::path dir = path.parent_path();

                if (!dir.empty() && !fs::exists(dir)) {
                    // Create directories recursively
                    try {
                        fs::create_directories(dir); // returns true/false
                        std::cout << "Directory exists or created: " << dir << "\n";
                    } catch (const std::exception& e) {
                        std::cerr << "Error creating directory: " << dir << " -> " << e.what() << "\n";
                    }
                }
            }
            
            bool file_exists(const std::string& filename) const {
                return fs::exists(filename);
            }

            void write_header(std::ofstream& file, const std::vector<CsvField>& extras = {}) const {
                file << "Hostname,Benchmark,Size,Internal loop iters,Num. Runs,Run. ID,Time [ms],Device energy [mj],Host energy [mj],Power trace,Frequency trace,Temperature trace";
                // Add custom entry in the .csv file
                for (const auto& f : extras)
                    file << "," << f.key;

                file << "\n";
            }

        public:
            Logger(const std::string& output_file_path)
                : output_file_path(output_file_path) {
                ensure_directory_exists();
            }
            
             // T define the data type used for the collective operation
            template <typename T>
            struct ProfilingInfo {
                std::string host_name;
                std::string bench_name;
                size_t size;
                int num_iters;
                int num_runs;
                int run_id; // index of the current run
                data_types::time_t time_ms; // time to solution of the rank to execute the specified collective in ms
                data_types::energy_t device_energy_mj; // in mj 
                data_types::energy_t host_energy_mj; // in mj
                data_types::power_trace_t power_trace;
                data_types::freq_trace_t freq_trace;
                data_types::temp_trace_t temp_trace;

                utils::data_types::GPUMode gpu_mode;
            };

            // The log interface take as input ProfilingInfo but we can also add other fields to the csv as ExtraFields...
            template<typename T, typename... ExtraFields>
            void log_result(const ProfilingInfo<T>& info, ExtraFields&&... extras){
               
                std::vector<CsvField> extra_fields{ std::forward<ExtraFields>(extras)... };

                std::string filename = output_file_path;
                
                // Only rank 0 write the csv header 
                bool is_new_file = !file_exists(filename);
                bool needs_header = is_new_file ? 1 : 0;

                if (needs_header) {
                    std::ofstream header_file(filename, std::ios::app);
                    if (header_file.is_open()) {
                        write_header(header_file, extra_fields);
                        header_file.close();
                    }
                }

                std::ofstream file(filename, std::ios::app);
                if (!file.is_open()) {
                    std::cerr << "Warning: Could not open log file: " << filename << std::endl;
                    return;
                }
                        

                    
                // Adde energy for each rank, then add also another line for avg time and energy for all the ranks
                // Scrivi la riga di dati
                file << info.host_name << ","
                     << info.bench_name << ","
                     << info.size << ","
                     << info.num_iters << ","
                     << info.num_runs << ","
                     << info.run_id << ","
                     << info.time_ms << ","
                     << info.device_energy_mj << ","
                     << info.host_energy_mj  << "," // can be 0 to num_ranks or aggregate
                     << utils::data_types::power_trace_to_string(info.power_trace) << ","
                     << utils::data_types::freq_trace_to_string(info.freq_trace) << ","
                     << utils::data_types::temp_trace_to_string(info.temp_trace);



                    // Write extra fields if any
                    for (const auto& f : extra_fields)
                        file << "," << f.value;
                    
                    file << "\n";
                    
                    file.close();
            }
          
            
            
            
         
            
            static void print_usage() {
                std::cout << "\nLogger Usage:" << std::endl;
                std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
                std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
                std::cout << "\nOutput format: CSV files with columns:" << std::endl;
                std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, gpu_mode, test_passed, time_ms" << std::endl;
            }


        };
    }
}