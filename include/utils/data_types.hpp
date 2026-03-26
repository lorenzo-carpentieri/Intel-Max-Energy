#pragma once

#include <string>
#include <vector>
#include <tuple>

namespace utils{
    namespace data_types{
        enum GPUMode {
            COMPOSITE,
            FLAT
        };
        using power_t = unsigned long long;
        using freq_t = unsigned int;
        using temperature_t = unsigned long long;
        using time_t = double;
        using energy_t = double;
        using timestamp_t = unsigned long long; 
        using power_trace_t = std::vector<std::tuple<timestamp_t,power_t>>;
        using freq_trace_t = std::vector<std::tuple<timestamp_t,freq_t>>;
        using temp_trace_t = std::vector<std::tuple<timestamp_t,temperature_t>>;


        inline std::string power_trace_to_string(const power_trace_t& trace)
            {
                std::ostringstream oss;
                oss << "\"[";

                for (std::size_t i = 0; i < trace.size(); ++i) {
                    const auto& [timestamp, power] = trace[i];
                    oss << "(" << timestamp << ";" << power << ")";
                }

                oss << "]\"";
                return oss.str();
        }

        inline std::string freq_trace_to_string(const freq_trace_t& trace)
        {
                std::ostringstream oss;
                oss << "\"[";

                for (std::size_t i = 0; i < trace.size(); ++i) {
                    const auto& [timestamp, power] = trace[i];
                    oss << "(" << timestamp << ";" << power << ")";
                }

                oss << "]\"";
                return oss.str();
        }

        inline std::string temp_trace_to_string(const temp_trace_t& trace)
        {
                std::ostringstream oss;
                oss << "\"[";

                for (std::size_t i = 0; i < trace.size(); ++i) {
                    const auto& [timestamp, temp] = trace[i];
                    oss << "(" << timestamp << ";" << temp << ")";
                }

                oss << "]\"";
                return oss.str();
        }
    }
}