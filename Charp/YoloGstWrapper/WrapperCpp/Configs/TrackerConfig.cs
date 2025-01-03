namespace WrapperCpp.Configs;

public class TrackerConfig
{
    /// <summary>
    /// Frame rate.
    /// </summary>
    public int FrameRate { get; set; }

    /// <summary>
    /// Track buffer.
    /// </summary>
    public int TrackBuffer { get; set; }
    
    /// <summary>
    /// Threshold.
    /// </summary>
    public float TrackThresh { get; set; }

    /// <summary>
    /// Treshold new object.
    /// </summary>
    public float HighThresh { get; set; }

    /// <summary>
    /// Math treshold.
    /// </summary>
    public float MathThresh { get; set; }

    /// <summary>
    /// Count tracks.
    /// </summary>
    public int MaxTrack { get; set; }
}