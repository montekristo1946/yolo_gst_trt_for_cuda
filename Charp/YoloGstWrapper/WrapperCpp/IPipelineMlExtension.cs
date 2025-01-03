using WrapperCpp.Dto;

namespace WrapperCpp;

/// <summary>
/// Dto rectangles.
/// </summary>
public record DoInferenceRectDetectResult()
{
    /// <summary>
    /// Rectangles.
    /// </summary>
    public RectDetect[] RectDetects { get; init; }

    /// <summary>
    /// Correct inference.
    /// </summary>
    public bool IsSuccess { get; init; }
}

/// <summary>
/// Dto images
/// </summary>
public record DoInferenceImageResult()
{
    /// <summary>
    /// Image in JPG.
    /// </summary>
    public byte[] ImageInJpeg { get; init; } = [];

    /// <summary>
    /// Correct inference.
    /// </summary>
    public bool IsSuccess { get; init; } = false;

    /// <summary>
    /// TimeStamp images.
    /// </summary>
    public ulong TimeStamp { get; init; } = 0;
}

/// <summary>
/// IPipelineMlExtension.
/// </summary>
public interface IPipelineMlExtension: IDisposable
{
    /// <summary>
    /// Get ml Rectangle Yolo.
    /// </summary>
    /// <returns></returns>
    public DoInferenceRectDetectResult DoInferencePipeline();

    /// <summary>
    /// Get Image in JPG.
    /// </summary>
    /// <returns></returns>
    public DoInferenceImageResult GetCurrenImage();
    
    /// <summary>
    /// 
    /// </summary>
    /// <param name="connetctionString"></param>
    /// <returns></returns>
    public bool StartPipelineGst(string connetctionString);
}