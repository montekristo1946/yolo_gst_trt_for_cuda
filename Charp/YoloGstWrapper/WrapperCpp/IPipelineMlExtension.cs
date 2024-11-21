using WrapperCpp.Dto;

namespace WrapperCpp;

public record DoInferenceRectDetectResult()
{
    
    public RectDetect [] RectDetects { get; init; }
    
    public bool IsSuccess { get; init; }
}

public record DoInferenceImageResult()
{

    public byte[] ImageInJpeg { get; init; } = [];

    public bool IsSuccess { get; init; } = false;
    
    public ulong TimeStamp { get; init; } = 0;
}

public interface IPipelineMlExtension
{
    /// <summary>
    /// Rectangle Yolo.
    /// </summary>
    /// <returns></returns>
    public DoInferenceRectDetectResult DoInferencePipeline();

    /// <summary>
    /// Image in JPG
    /// </summary>
    /// <returns></returns>
    public DoInferenceImageResult GetCurrenImage();

}