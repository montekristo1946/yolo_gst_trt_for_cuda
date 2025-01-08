using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using WrapperCpp.InfrastructureCPP.PInvokeDto;

namespace WrapperCpp.InfrastructureCPP;

internal static class PipelinePInvoke
{
    // private const string _patchDll = @"./LibsCPP/libExtensionCharp.so";
    private const string _patchDll =
    @"/mnt/Disk_C/git/yolo_gst_trt_for_cuda/CPP/cmake-build-release/libExtensionCharp.so";

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "InitLogger", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void InitLogger(StringBuilder logPathFile);


    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateCudaStream", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateCudaStream();

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateTRTEngine", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateTRTEngine(ref TRTEngineConfig config, IntPtr cudaStream);


    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateBufferFrameGpu", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateBufferFrameGpu();


    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateGstBufferManager", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateGstBufferManager(IntPtr bufferFrameGpu, IntPtr cudaStream);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateGstDecoder", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateGstDecoder(IntPtr bufferManager);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateNvJpgEncoder", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateNvJpgEncoder(IntPtr cudaStream);
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateEnginPipeline", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateEnginPipeline(
        IntPtr trtEngine,
        IntPtr bufferFrameGpu,
        IntPtr cudaStream,
        ref SettingPipeline configPipeline,
        IntPtr encoder,
        IntPtr trackerManager,
        IntPtr algorithmsPolygon);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "DoInferencePipeline", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool DoInferencePipeline(IntPtr _pipeline, ref PipelineOutputData outputData);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "Dispose", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool Dispose(IntPtr disposeObj);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "DisposeArrChar", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool DisposeArrChar(IntPtr disposeObj);


    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "DisposeRectDetectExternal", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool DisposeRectDetectExternal(IntPtr ptr);
  
    
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "GetCurrenImage", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool GetCurrenImage(IntPtr pipeline, ref ImageFrame imageFrame);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "ConverterNetworkWeight", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool ConverterNetworkWeight(
        StringBuilder pathOnnxModelChar,
        StringBuilder exportPathModelChar,
        ref LayerSize config,
        int idGpu,
        bool setHalfModel = true
    );

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "StartPipelineGst", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool StartPipelineGst(IntPtr gstDecoder, StringBuilder connectionOnCameraPipliene);


    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateTrackerManager", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateTrackerManager(
        int frameRate,
        int trackBuffer,
        float trackThresh,
        float highThresh,
        float matchThresh,
        int maxNumTrackers);
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "CreateAlgorithmsPolygon", CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr CreateAlgorithmsPolygon();   

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "AlgorithmsPolygonClear", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool AlgorithmsPolygonClear(IntPtr algorithmsPolygon);   
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "AlgorithmsPolygonAppend", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool AlgorithmsPolygonAppend(IntPtr algorithmsPolygon, ref PolygonsSettingsExternal polygons);

  
}