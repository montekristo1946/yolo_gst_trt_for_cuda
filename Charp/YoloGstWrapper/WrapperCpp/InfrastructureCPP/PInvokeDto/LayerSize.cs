using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct LayerSize
{
    [FieldOffset(0)]
    private readonly int BatchSize;

    [FieldOffset(4)]
    private readonly int Channel;

    [FieldOffset(8)]
    private readonly int Width;

    [FieldOffset(12)]
    private readonly int Height;
    internal LayerSize(
        int batchSize,
        int channel,
        int width,
        int height
    )
    {
        BatchSize = batchSize;
        Channel   = channel;
        Width     = width;
        Height    = height;
    }
}
