#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <stdexcept>
namespace utils{
    namespace arg_parse {

        using value_t = std::string;
        using map_t   = std::unordered_map<std::string, std::string>;

        // Known numeric parameters
        inline const std::unordered_set<std::string> known_size_params = {
            "size",
            "iters",
            "runs",
            "csv-path",
            "freq0",
            "freq1"
            "kernel1", // comp or mem
            "kernel2"
        };

        // Check if argument starts with "--"
        inline bool is_param(const std::string& s) {
            return s.size() > 2 && s.rfind("--", 0) == 0;
        }

        // Strip leading "--"
        inline std::string strip_prefix(const std::string& s) {
            return s.substr(2);
        }

        // Parse key=value format
        inline map_t parse(int argc, char** argv) {
            map_t params;

            for (int i = 1; i < argc; ++i) {
                std::string arg = argv[i];

                if (!is_param(arg)) {
                    throw std::runtime_error("Unexpected argument: " + arg);
                }

                std::string keyval = strip_prefix(arg);
                auto pos = keyval.find('=');
                if (pos == std::string::npos) {
                    throw std::runtime_error("Expected --key=value format, got: " + arg);
                }

                std::string key = keyval.substr(0, pos);
                std::string value = keyval.substr(pos + 1);

                if (known_size_params.count(key)) {
                    try {
                        params[key] = value;
                    } catch (...) {
                        throw std::runtime_error("Invalid numeric value for parameter: " + key);
                    }
                } else {
                    params[key] = value; // unknown parameter stored as string
                }
            }

            return params;
        }

    } // namespace arg_parse
}