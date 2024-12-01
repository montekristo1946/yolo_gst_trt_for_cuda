#include "BuilderOnnx.hpp"


bool BuilderOnnx::ApiToFileModel(string &modelInputPath, string &modelOutputPath,bool setHalfModel) {

    if (modelInputPath.empty() || modelOutputPath.empty()) {
        _logger->error("[BuilderOnnx::ApiToFileModel] wtsName.empty() ||exportSave.empty()");
        return false;
    }

    NvidiaLogger nvidiaLogger;
    nvidiaLogger.log(ILogger::Severity::kINFO, "[BuilderOnnx::ApiToFileModel] Init LoggerNvidia ok");

    TRTUniquePtr<IBuilder> builder{createInferBuilder(nvidiaLogger)};
    if (builder == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] builder == Null ");
        return false;
    }

    TRTUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
    if (config == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] config == Null ");
        return false;
    }

    const auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<INetworkDefinition> network{builder->createNetworkV2(flags)};
    if (network == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] network == Null ");
        return false;
    }

    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, nvidiaLogger)};
    if (parser == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] parser == Null ");
        return false;
    }

    bool parsingSuccess = parser->parseFromFile(modelInputPath.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    if (!parsingSuccess) {
        _logger->error("[BuilderOnnx::ApiToFileModel] Failed to parse model ");
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] profile == Null ");
        return false;
    }

    profile->setDimensions(_nameInputLayer.c_str(), OptProfileSelector::kMIN, Dims4{_minimalBatch, _channel, _inputH, _inputW});
    profile->setDimensions(_nameInputLayer.c_str(), OptProfileSelector::kOPT, Dims4{_optimalBatch, _channel, _inputH, _inputW});
    profile->setDimensions(_nameInputLayer.c_str(), OptProfileSelector::kMAX, Dims4{_maxBatch, _channel, _inputH, _inputW});
    config->addOptimizationProfile(profile);

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 128 * (1 << 20));// 128 MiB

    if (setHalfModel) {
        config->setFlag(BuilderFlag::kFP16);
    }

    TRTUniquePtr<IHostMemory> modelStream{builder->buildSerializedNetwork(*network,*config)};
    if (modelStream == nullptr) {
        _logger->error("[BuilderOnnx::ApiToFileModel] modelStream == Null ");
        return false;
    }

    ofstream file(modelOutputPath, ios::binary);
    if (!file) {
        _logger->error("[BuilderOnnx::ApiToFileModel] could not open plan output file");
        return false;
    }
    file.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    file.close();

    return true;
}
