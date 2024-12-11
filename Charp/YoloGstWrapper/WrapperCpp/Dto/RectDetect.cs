using WrapperCpp.InfrastructureCPP.PInvokeDto;

namespace WrapperCpp.Dto;

public record RectDetect()
{
    /// <summary>
    ///     Center Rectangle X
    /// </summary>
    public float X { get; init; } = -1F;

    /// <summary>
    ///     Center Rectangle Y
    /// </summary>
    public float Y { get; init; } = -1F;

    /// <summary>
    ///     Width  Rectangle
    /// </summary>
    public float Width { get; init; } = -1F;

    /// <summary>
    ///     Height  Rectangle
    /// </summary>
    public float Height { get; init; } = -1F;

    /// <summary>
    ///     Id Labels in ML
    /// </summary>
    public int IdClass { get; init; } = -1;

    /// <summary>
    ///     Veracity this rectangle
    /// </summary>
    public float Veracity { get; init; } = -1F;
    
    /// <summary>
    ///     TimeStamp images
    /// </summary>
    public uint TimeStamp { get; init; } = 0;
    
    public int TrackId { get; set; }= -1;

    public static RectDetect RectDetectEnginToRectDetect(RectDetectEngin rectDetectEngin)
    {
        return new RectDetect()
        {
            X = rectDetectEngin.X,
            Y = rectDetectEngin.Y,
            Width = rectDetectEngin.Width,
            Height = rectDetectEngin.Height,
            IdClass = rectDetectEngin.IdClass,
            Veracity = rectDetectEngin.Veracity,
            TimeStamp = rectDetectEngin.TimeStamp,
            TrackId = rectDetectEngin.TrackId
        };
    }
}
