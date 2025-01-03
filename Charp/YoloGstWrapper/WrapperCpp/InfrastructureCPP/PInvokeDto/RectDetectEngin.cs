using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
public struct RectDetectEngin
{
    [FieldOffset(0)]
    public float X;

    [FieldOffset(4)]
    public float Y;

    [FieldOffset(8)]
    public float Width;

    [FieldOffset(12)]
    public float Height;

    [FieldOffset(16)]
    public int IdClass;

    [FieldOffset(20)]
    public float Veracity;
    
    [FieldOffset(24)]
    public int TrackId;
    
    [FieldOffset(28)]
    public uint TimeStamp;

    [FieldOffset(32)]
    public int PolygonId;

    
}
