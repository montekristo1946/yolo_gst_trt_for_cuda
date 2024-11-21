#ifndef TENSORRTTOOLS_MainLogger_H
#define TENSORRTTOOLS_MainLogger_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/common.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace std;
using namespace spdlog;

class MainLogger {
public:
    MainLogger(string pathToLogFile, int maxSizeMbyte = 10, int maxFiles = 10, string nameLogger = "MainLogger") {
        auto console_sink = make_shared<sinks::stdout_color_sink_mt>();
        console_sink->set_color_mode(color_mode::always);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [thd %t] %v");
        console_sink->set_level(level::debug);

        auto maxSize = 1048576 * maxSizeMbyte;//10 mb
        auto file_sink = make_shared<sinks::rotating_file_sink_mt>(pathToLogFile, maxSize, maxFiles);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [thd %t] %v");
        file_sink->set_level(level::debug);

        set_default_logger(make_shared<logger>(nameLogger, sinks_init_list({console_sink, file_sink})));
    }

    // static spdlog::logger logger("my_logger");
};


#endif //TENSORRTTOOLS_MainLogger_H
