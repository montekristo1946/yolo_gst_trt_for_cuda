using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct TRTEngineConfig
{
    [FieldOffset(0)]
    private readonly string EngineName;

    [FieldOffset(8)]
    private readonly int DeviseId;
    
    [FieldOffset(12)]
    private readonly float ConfThresh;

    [FieldOffset(16)]
    private readonly float NmsThresh;
    
    [FieldOffset(20)]
    private readonly int MaxNumOutputBbox;
    
    internal TRTEngineConfig(
        string engineName,
        int deviseId, float confThresh, float nmsThresh, int maxNumOutputBbox)
    {
        EngineName = engineName;
        DeviseId   = deviseId;
        ConfThresh = confThresh;
        NmsThresh = nmsThresh;
        MaxNumOutputBbox = maxNumOutputBbox;
    }
}
