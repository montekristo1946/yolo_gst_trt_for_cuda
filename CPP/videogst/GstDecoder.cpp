#include "include/GstDecoder.h"


GstDecoder::GstDecoder(GstBufferManager* gstBufferManager)
{
    if (gstBufferManager == nullptr)
        throw std::runtime_error("Null reference exception {name: GstBufferManager}");

    bufferManager = gstBufferManager;
    _isStreaming = false;
    _isEOS = false;
}

GstDecoder::~GstDecoder()
{
    _logger->info("[~GstDecoder] Call");
    Close();

    if (_appSink != NULL)
    {
        gst_object_unref(_appSink);
        _appSink = NULL;
    }

    if (_bus != NULL)
    {
        gst_object_unref(_bus);
        _bus = NULL;
    }

    if (_pipeline != NULL)
    {
        gst_object_unref(_pipeline);
        _pipeline = NULL;
    }
}


bool GstDecoder::InitPipeline(string basicString)
{
    try
    {
        _logger->info("[GstDecoder::InitPipeline] Start init");

        int argc = 0;
        if (!gst_init_check(&argc, NULL, NULL))
        {
            _logger->error("[GstDecoder::InitPipeline] failed to initialize gstreamer library with gst_init()");
            return false;
        }


        uint32_t ver[] = {0, 0, 0, 0};
        gst_version(&ver[0], &ver[1], &ver[2], &ver[3]);
        _logger->info("initialized gstreamer, version {}.{}.{}.{}", (int)ver[0], (int)ver[1], (int)ver[2],
                      (int)ver[3]);

        gst_debug_remove_log_function(gst_debug_log_default);
        gst_debug_add_log_function(RilogDebugFunction, NULL, NULL);

        gst_debug_set_active(true);
        gst_debug_set_colored(false);
        gst_debug_set_default_threshold(GST_LEVEL_ERROR);

        GError* err = NULL;
        _pipeline = gst_parse_launch(basicString.c_str(), &err);

        if (err != NULL)
        {
            _logger->error("[GstDecoder::InitPipeline] failed to create pipeline: {}", err->message);
            g_error_free(err);
            return false;
        }

        GstPipeline* pipeline = GST_PIPELINE(_pipeline);

        if (!pipeline)
        {
            _logger->error("[GstDecoder::InitPipeline] failed to cast GstElement into GstPipeline");
            return false;
        }

        _bus = gst_pipeline_get_bus(pipeline);

        if (!_bus)
        {
            _logger->error("[GstDecoder::InitPipeline] failed to retrieve GstBus from pipeline");
            return false;
        }

        GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
        GstAppSink* appsink = GST_APP_SINK(appsinkElement);

        if (!appsinkElement || !appsink)
        {
            _logger->error("[GstDecoder::InitPipeline] failed to retrieve AppSink element from pipeline");
            return false;
        }

        _appSink = appsink;

        // setup callbacks
        GstAppSinkCallbacks cb;
        memset(&cb, 0, sizeof(GstAppSinkCallbacks));
        cb.eos = OnEOS;
        cb.new_preroll = OnPreroll; // disabled b/c preroll sometimes occurs during Close() and crashes
        cb.new_sample = OnBuffer;
        gst_app_sink_set_callbacks(_appSink, &cb, (void*)this, NULL);

        return true;
    }
    catch (...)
    {
        _logger->error("[GstDecoder::InitPipeline] Unknown exception!");
    }


    return false;
}

void GstDecoder::RilogDebugFunction(GstDebugCategory* category, GstDebugLevel level,
                                    const gchar* file, const char* function,
                                    gint line, GObject* object, GstDebugMessage* message,
                                    gpointer data)
{
    const char* typeName = " ";

    if (object != NULL)
    {
        typeName = G_OBJECT_TYPE_NAME(object);
    }

    std::string formattedString = fmt::format(
        "[level:{}] [type:{}] [category:{}] [file:{}] [line:{}] [function:{}] [message:{}]",
        GetAlarmLevel(level),
        typeName,
        gst_debug_category_get_name(category),
        file,
        line,
        function,
        gst_debug_message_get(message));

    spdlog::info("[GstDecoder::RilogDebugFunction] " + formattedString);
}

void GstDecoder::OnEOS(_GstAppSink* sink, void* user_data)
{
    warn("[GstDecoder::OnEOS] end of stream (EOS)");

    if (!user_data)
        return;

    GstDecoder* dec = (GstDecoder*)user_data;

    dec->_isEOS = true;
    dec->_isStreaming = false;
}

GstFlowReturn GstDecoder::OnPreroll(_GstAppSink* sink, void* user_data)
{
    info("[GstDecoder::OnPreroll] onPreroll()");

    if (!user_data)
        return GST_FLOW_OK;

    GstDecoder* dec = (GstDecoder*)user_data;


    // onPreroll gets called sometimes, just pull and free the buffer
    // otherwise the pipeline may hang during shutdown
    GstSample* gstSample = gst_app_sink_pull_preroll(dec->_appSink);

    if (!gstSample)
    {
        return GST_FLOW_OK;
    }

    gst_sample_unref(gstSample);

    dec->CheckMsgBus();
    return GST_FLOW_OK;
}

GstFlowReturn GstDecoder::OnBuffer(_GstAppSink* sink, void* user_data)
{
    if (!user_data)
        return GST_FLOW_OK;

    GstDecoder* dec = (GstDecoder*)user_data;

    dec->CheckBuffer();
    dec->CheckMsgBus();

    return GST_FLOW_OK;
}

void GstDecoder::CheckMsgBus()
{
    while (true)
    {
        GstMessage* msg = gst_bus_pop(_bus);

        if (!msg)
            break;

        GstMessagePrint(_bus, msg, this);
        gst_message_unref(msg);
    }
}

void GstDecoder::CheckBuffer()
{
    if (!_appSink)
        return;
    // block waiting for the sample
    GstSample* gstSample = gst_app_sink_pull_sample(_appSink);

    if (!gstSample)
    {
        warn("[GstDecoder::CheckBuffer] app_sink_pull_sample() returned NULL");
        return;
    }

    // retrieve sample caps
    GstCaps* gstCaps = gst_sample_get_caps(gstSample);

    if (!gstCaps)
    {
        warn("[GstDecoder::CheckBuffer] gst_sample had NULL caps...");
        release_return;
    }


    GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);

    if (!gstBuffer)
    {
        warn("[GstDecoder::CheckBuffer] app_sink_pull_sample() returned NULL...");
        release_return;
    }

    if (!bufferManager->Enqueue(gstBuffer, gstCaps))
    {
        warn("[GstDecoder::CheckBuffer] failed to handle incoming buffer");
        release_return;
    }

    release_return;
}

std::string GstDecoder::GetAlarmLevel(GstDebugLevel level)
{
    switch (level)
    {
    case GST_LEVEL_NONE: return "NONE";
    case GST_LEVEL_ERROR: return "ERROR";
    case GST_LEVEL_WARNING: return "WARNING";
    case GST_LEVEL_FIXME: return "FIXME";
    case GST_LEVEL_INFO: return "INFO";
    case GST_LEVEL_DEBUG: return "DEBUG";
    case GST_LEVEL_LOG: return "LOG";
    case GST_LEVEL_TRACE: return "TRACE";
    default: return "UNKNOWN";
    }
}

void GstDecoder::Close()
{
    if (!_isStreaming)
        return;

    info("[GstDecoder::Close] stopping pipeline, transitioning to GST_STATE_NUL");

    const GstStateChangeReturn result = gst_element_set_state(_pipeline, GST_STATE_NULL);

    if (result != GST_STATE_CHANGE_SUCCESS)
    {
        error("[GstDecoder::Close] failed to set pipeline state to PLAYING (error {})");
        return;
    }
    this_thread::sleep_for(std::chrono::microseconds(250 * 1000));
    CheckBuffer();
    _isStreaming = false;

    info("[GstDecoder::Close] pipeline stopped");
}

bool GstDecoder::Open()
{
    if (_isStreaming)
        return true;

    // transition pipline to STATE_PLAYING
    info("[GstDecoder::Open] opening gstDecoder for streaming, transitioning pipeline to GST_STATE_PLAYING");

    const GstStateChangeReturn result = gst_element_set_state(_pipeline, GST_STATE_PLAYING);

    if (result == GST_STATE_CHANGE_FAILURE || result == GST_STATE_CHANGE_NO_PREROLL)
    {
        info("[GstDecoder::Open] failed to set pipeline state to PLAYING ");
        return false;
    }

    CheckMsgBus();
    usleep(100 * 1000);
    CheckMsgBus();

    _isStreaming = true;
    return true;
}
