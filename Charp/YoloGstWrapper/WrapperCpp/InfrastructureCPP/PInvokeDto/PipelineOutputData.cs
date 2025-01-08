using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct PipelineOutputData
{

    [FieldOffset(0)]
    internal  IntPtr Rectangles;

    [FieldOffset(8)]
    internal  uint RectanglesLen;
    
    [FieldOffset(12)]
    internal  uint StepStructure;

}
