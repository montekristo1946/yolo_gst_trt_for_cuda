
#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/common.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>

#if NV_TENSORRT_MAJOR >= 10
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

using namespace std;
using namespace nvinfer1;
using namespace spdlog;


class NvidiaLogger : public ILogger {
public:
    NvidiaLogger(Severity severity = Severity::kINFO, string nameLogger = "MainLogger")
            : mReportableSeverity(severity) {
        _handler = spdlog::get(nameLogger);
    }

    void log(Severity severity, const char *msg) TRT_NOEXCEPT override {
        if (_handler == nullptr) {
            error("[NvidiaLogger] Not init, error:{}", msg);
        }

        if ((severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)) {
            _handler->error(msg);
            return;
        }

        if (severity == Severity::kWARNING) {
            _handler->warn(msg);
            return;
        }
        _handler->info(msg);
    }

private:
    shared_ptr<logger> _handler = nullptr;
    Severity mReportableSeverity;
};

#endif // TENSORRT_LOGGING_H
