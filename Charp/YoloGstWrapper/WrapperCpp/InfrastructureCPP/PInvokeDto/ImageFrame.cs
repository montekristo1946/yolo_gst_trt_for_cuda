using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 0)]
internal struct ImageFrame
{
    [FieldOffset(0)]
    internal readonly IntPtr ImagesData;

    [FieldOffset(8)]
    internal readonly int ImageLen;
    
    [FieldOffset(16)]
    public readonly ulong  TimeStamp;
    
}