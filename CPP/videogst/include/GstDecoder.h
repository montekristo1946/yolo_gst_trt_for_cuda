
#ifndef __GSTREAMER_DECODER_H__
#define __GSTREAMER_DECODER_H__

#include <BufferFrameGpu.h>

#include "IDispose.h"
#include "gstelement.h"
#include <gstparse.h>
#include "gstinfo.h"
#include "gstpipeline.h"
#include "gstappsink.h"
#include "GstUtility.h"
#include "gstsample.h"
#include "GstBufferManager.h"
#include "gst.h"

#define release_return { gst_sample_unref(gstSample); return; }

class GstDecoder : public IDispose
{
public:
    GstDecoder(GstBufferManager* gstBufferManager);

    ~GstDecoder();


    bool StartPipeline( string connectCamera );

protected:

    bool InitPipeline(string basicString);
    bool Open();

    GstElement* _pipeline;
    GstBus* _bus;
    _GstAppSink* _appSink;
    static void RilogDebugFunction(GstDebugCategory* category, GstDebugLevel level,
                                   const gchar* file, const char* function,
                                   gint line, GObject* object, GstDebugMessage* message,
                                   gpointer data);

    static void OnEOS(_GstAppSink* sink, void* user_data);
    static GstFlowReturn OnPreroll(_GstAppSink* sink, void* user_data);
    static GstFlowReturn OnBuffer(_GstAppSink* sink, void* user_data);

    bool _isEOS;
    bool _isStreaming;
    GstBufferManager* _bufferManager;

    void CheckMsgBus();

    void CheckBuffer();

    static std::string GetAlarmLevel(GstDebugLevel level);

    void Close();

    shared_ptr<logger> _logger = get("MainLogger");
};

#endif
