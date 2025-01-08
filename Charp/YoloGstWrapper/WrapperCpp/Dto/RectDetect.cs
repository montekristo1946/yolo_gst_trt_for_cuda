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
    
    /// <summary>
    ///     Track object in images
    /// </summary>
    public int TrackId { get; init; }= -1;

    /// <summary>
    ///     Polygon id
    /// </summary>
    public int[] PolygonsId = [];

    public static RectDetect RectDetectEnginToRectDetect(RectDetectExternal rectDetectExternal, int [] polygonsId)
    {
        return new RectDetect()
        {
            X = rectDetectExternal.X,
            Y = rectDetectExternal.Y,
            Width = rectDetectExternal.Width,
            Height = rectDetectExternal.Height,
            IdClass = rectDetectExternal.IdClass,
            Veracity = rectDetectExternal.Veracity,
            TimeStamp = rectDetectExternal.TimeStamp,
            TrackId = rectDetectExternal.TrackId,
            PolygonsId = polygonsId
        };
    }
}
