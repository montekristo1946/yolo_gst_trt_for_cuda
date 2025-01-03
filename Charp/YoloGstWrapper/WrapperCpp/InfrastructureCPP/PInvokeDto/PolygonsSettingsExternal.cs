using System.Runtime.InteropServices;
using WrapperCpp.Configs;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct PolygonsSettingsExternal
{
    [FieldOffset(0)] 
    internal  int IdPolygon;
    
    [FieldOffset(8)] 
    internal unsafe float* PolygonsX;

    [FieldOffset(16)] 
    internal unsafe float* PolygonsY;
    
    [FieldOffset(24)] 
    internal  uint CountPoints;
    
}
