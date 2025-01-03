namespace WrapperCpp.Configs;

public class YoloConfigs()
{
    /// <summary>
    ///  Path to engine.
    /// </summary>
    public string EnginePath { get; set; }

    /// <summary>
    ///  Gpu id.
    /// </summary>
    public int DeviseId { get; set; }

    /// <summary>
    /// Yolo thresh value.
    /// </summary>
    public float ConfThresh { get; set; }

    /// <summary>
    /// Nms thresh value.
    /// </summary>
    public float NmsThresh { get; set; }

    /// <summary>
    /// Max src num output bbox. default 1000.
    /// </summary>
    public int MaxNumOutputBbox { get; set; }

    /// <summary>
    /// width of input Yolo image.
    /// </summary>
    public int WidthImgMl { get; set; }

    /// <summary>
    /// Height of input Yolo image.
    /// </summary>
    public int HeightImgMl { get; set; }

    /// <summary>
    /// Count img to background defalt 25.
    /// </summary>
    public int CountImgToBackground { get; set; }

    /// <summary>
    /// GST connection string pipeline.
    /// </summary>
    // public string ConnetctionString { get; set; }

    /// <summary>
    /// Path to log file.
    /// </summary>
    public string PathLogFile { get; set; }
};