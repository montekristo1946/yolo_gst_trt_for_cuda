namespace WrapperCpp.Configs;

public class PolygonWrapCpp
{

    /// <summary>
    /// ID.
    /// </summary>
    public int Id { get; init; }

    /// <summary>
    /// Polygons Points.
    /// </summary>
    public PointWrapCpp[] Points { get; init; } = [];
    
}