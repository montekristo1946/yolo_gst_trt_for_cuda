#include "GstUtility.h"


using namespace spdlog;


gboolean GstMessagePrint(GstBus* bus, GstMessage* message, gpointer user_data)
{
    switch (GST_MESSAGE_TYPE(message))
    {
    case GST_MESSAGE_ERROR:
        {
            GError* err = NULL;
            gchar* dbg_info = NULL;

            gst_message_parse_error(message, &err, &dbg_info);

            error("gstreamer ERROR {}", err->message);
            error("gstreamer Debugging info: {}", (dbg_info) ? dbg_info : "none");

            g_error_free(err);
            g_free(dbg_info);
            break;
        }
    case GST_MESSAGE_EOS:
        {
            info("gstreamer EOS signal... {}", GST_OBJECT_NAME(message->src));
            break;
        }
    case GST_MESSAGE_STATE_CHANGED:
        {
            GstState old_state, new_state;

            gst_message_parse_state_changed(message, &old_state, &new_state, NULL);

            info("gstreamer changed state from {} to {} ==> {}",
                 gst_element_state_get_name(old_state), gst_element_state_get_name(new_state),
                 GST_OBJECT_NAME(message->src));
            break;
        }
    case GST_MESSAGE_STREAM_STATUS:
        {
            GstStreamStatusType streamStatus;
            gst_message_parse_stream_status(message, &streamStatus, NULL);

            info("gstreamer stream status {} ==> {}", GstStreamStatusString(streamStatus),
                 GST_OBJECT_NAME(message->src));
            break;
        }
    case GST_MESSAGE_TAG:
        {
            GstTagList* tags = NULL;
            gst_message_parse_tag(message, &tags);
            gchar* txt = gst_tag_list_to_string(tags);

            if (txt != NULL)
            {
                info("gstreamer [GST_MESSAGE_TAG] {} {}", GST_OBJECT_NAME(message->src), txt);
                g_free(txt);
            }

            if (tags != NULL)
                gst_tag_list_free(tags);

            break;
        }
    default:
        {
            info("gstreamer message {}  ==> {}", gst_message_type_get_name(GST_MESSAGE_TYPE(message)),
                 GST_OBJECT_NAME(message->src));
            break;
        }
    }

    return TRUE;
}

static const char* GstStreamStatusString(GstStreamStatusType status)
{
    switch (status)
    {
    case GST_STREAM_STATUS_TYPE_CREATE: return "CREATE";
    case GST_STREAM_STATUS_TYPE_ENTER: return "ENTER";
    case GST_STREAM_STATUS_TYPE_LEAVE: return "LEAVE";
    case GST_STREAM_STATUS_TYPE_DESTROY: return "DESTROY";
    case GST_STREAM_STATUS_TYPE_START: return "START";
    case GST_STREAM_STATUS_TYPE_PAUSE: return "PAUSE";
    case GST_STREAM_STATUS_TYPE_STOP: return "STOP";
    default: return "UNKNOWN";
    }
}
