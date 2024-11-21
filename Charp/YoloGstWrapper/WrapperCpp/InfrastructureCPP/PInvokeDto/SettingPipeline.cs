using System.Runtime.InteropServices;

namespace WrapperCpp.InfrastructureCPP.PInvokeDto;

[StructLayout(LayoutKind.Explicit, Pack = 1)]
internal struct SettingPipeline
{
    [FieldOffset(0)]
    private readonly int WidthImgMl;
    
    [FieldOffset(4)]
    private readonly int HeightImgMl;

    [FieldOffset(8)]
    private readonly int CountImgToBackground;


    public SettingPipeline(int widthImgMl, int heightImgMl, int countImgToBackground)
    {
        WidthImgMl = widthImgMl;
        HeightImgMl = heightImgMl;
        CountImgToBackground = countImgToBackground;
    }
}