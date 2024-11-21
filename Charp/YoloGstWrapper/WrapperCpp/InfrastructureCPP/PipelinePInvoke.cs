using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using WrapperCpp.InfrastructureCPP.PInvokeDto;

namespace WrapperCpp.InfrastructureCPP;

internal static class  PipelinePInvoke
{
    // private const string _patchDll = @"./LibsCPP/libExtensionCharpTemalCamera.so";
    private const string _patchDll = @"/mnt/Disk_C/git/yolo_gst_trt_for_cuda/CPP/cmake-build-release/libExtensionCharpTemalCamera.so";
    
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
        IntPtr gstBufferManager,
        IntPtr gstDecoder,
        IntPtr cudaStream,
        ref SettingPipeline configPipeline,
        IntPtr encoder,
        StringBuilder connetctionString);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "DoInferencePipeline", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool DoInferencePipeline(IntPtr _pipeline, ref PipelineOutputData outputData);
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "Dispose", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool Dispose(IntPtr disposeObj);
    
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "DisposeArr", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool DisposeArr(IntPtr disposeObj);
  
    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "GetCurrenImage", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool GetCurrenImage(IntPtr _pipeline, ref ImageFrame imageFrame);

    [SuppressUnmanagedCodeSecurity]
    [DllImport(_patchDll, EntryPoint = "ConverterNetworkWeight", CallingConvention = CallingConvention.Cdecl)]
    internal static extern bool ConverterNetworkWeight(
        StringBuilder pathOnnxModelChar,
        StringBuilder exportPathModelChar,
        ref LayerSize config,   
        int idGpu,
        bool setHalfModel = true
    );
}