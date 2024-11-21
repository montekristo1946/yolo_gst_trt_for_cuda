using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct PipelineOutputData
{

    [FieldOffset(0)]
    internal  unsafe RectDetectEngin* Rectangles;

    [FieldOffset(8)]
    internal  uint RectanglesLen;

}
