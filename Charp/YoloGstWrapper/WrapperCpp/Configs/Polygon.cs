namespace WrapperCpp.Configs;

public class Polygon
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