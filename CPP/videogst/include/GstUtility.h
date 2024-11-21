#ifndef YOLOGSTFORCUDA_GSTUTILITY_H
#define YOLOGSTFORCUDA_GSTUTILITY_H

#include <gst/gst.h>
#include "stdio.h"
#include <string>
#include <spdlog/spdlog.h>
#include <gstinfo.h>

static const char* GstStreamStatusString( GstStreamStatusType status );

gboolean GstMessagePrint(_GstBus* bus, _GstMessage* message, void* user_data);



#endif
