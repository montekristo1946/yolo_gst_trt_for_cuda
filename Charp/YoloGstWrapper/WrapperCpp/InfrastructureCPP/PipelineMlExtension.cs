using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Text;
using WrapperCpp.Configs;
using WrapperCpp.Dto;
using WrapperCpp.InfrastructureCPP.PInvokeDto;

namespace WrapperCpp.InfrastructureCPP;

using Serilog;

public class PipelineMlExtension : IDisposable, IPipelineMlExtension
{
    private readonly nint _cudaStream = nint.Zero;
    private readonly nint _trtEngine = nint.Zero;
    private readonly nint _bufferFrameGpu = nint.Zero;
    private readonly nint _bufferManager = nint.Zero;
    private nint _gstDecoder = nint.Zero;
    private readonly nint _encoder = nint.Zero;
    private readonly nint _pipeline = nint.Zero;
    private readonly nint _trackerManager = nint.Zero;

    private readonly ILogger _logger = Log.ForContext<PipelineMlExtension>();

    public PipelineMlExtension(YoloConfigs configYolo, TrackerConfig configTracker, bool isMockCppMl = false)
    {
        if (isMockCppMl)
        {
            _logger.Warning("[PipelineMlExtension] Not Init ML trt. This run mock ");

            return;
        }

        ArgumentNullException.ThrowIfNull(configYolo);
        ArgumentNullException.ThrowIfNull(configTracker);


        CreateCppLogger(configYolo.PathLogFile);

        _cudaStream = CreateCudaStream();
        _trtEngine = CreateTrtEngin(configYolo);
        _bufferFrameGpu = CreateBufferFrameGpu();
        _bufferManager = CreateGstBufferManager();
        _encoder = CreateEncoder();
        _trackerManager = CreateTrackerManager(configTracker);
        _pipeline = CreatePipeline(configYolo);
       
    }

    private IntPtr CreateTrackerManager(TrackerConfig configTracker)
    {
        if (configTracker.FrameRate < 0 || configTracker.FrameRate > 100 ||
            configTracker.TrackBuffer < 0 || configTracker.TrackBuffer > 100 ||
            configTracker.TrackThresh < 0 || configTracker.TrackThresh > 1 ||
            configTracker.HighThresh < 0 || configTracker.HighThresh > 1 ||
            configTracker.MathThresh < 0 || configTracker.MathThresh > 1 ||
            configTracker.MaxTrack < 0 || configTracker.MaxTrack > 10)
            throw new Exception("[PipelineMlExtension:CreateTarckerManager] configTracker not init");

        var retPtr = PipelinePInvoke.CreateTrackerManager(
            configTracker.FrameRate,
            configTracker.TrackBuffer,
            configTracker.TrackThresh,
            configTracker.HighThresh,
            configTracker.MathThresh,
            configTracker.MaxTrack);

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateTarckerManager] retPtr not init");
        }

        return retPtr;
    }

    private IntPtr CreatePipeline(YoloConfigs config)
    {
        var configPipeline = new SettingPipeline(config.WidthImgMl,
            config.HeightImgMl,
            config.CountImgToBackground);

        var retPtr = PipelinePInvoke.CreateEnginPipeline(
            _trtEngine,
            _bufferFrameGpu,
            _cudaStream,
            ref configPipeline,
            _encoder,
            _trackerManager);

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreatePipeline] retPtr not init");
        }

        return retPtr;
    }


    private IntPtr CreateEncoder()
    {
        var retPtr = PipelinePInvoke.CreateNvJpgEncoder(_cudaStream);

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateEncoder] retPtr not init");
        }

        return retPtr;
    }

    private IntPtr CreateGstDecoder()
    {
        if (_bufferManager == IntPtr.Zero)
            throw new Exception("[PipelineMlExtension:CreateGstDecoder] bufferManager not init");

        var retPtr = PipelinePInvoke.CreateGstDecoder(_bufferManager);

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateGstDecoder] retPtr not init");
        }

        return retPtr;
    }

    private IntPtr CreateGstBufferManager()
    {
        if (_bufferFrameGpu == IntPtr.Zero || _cudaStream == IntPtr.Zero)
            throw new Exception("[PipelineMlExtension:CreateGstBufferManager] bufferManager not init");

        var retPtr = PipelinePInvoke.CreateGstBufferManager(_bufferFrameGpu, _cudaStream);

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateGstBufferManager] retPtr not init");
        }

        return retPtr;
    }

    private IntPtr CreateBufferFrameGpu()
    {
        var retPtr = PipelinePInvoke.CreateBufferFrameGpu();

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateBufferFrameGpu] retPtr not init");
        }

        return retPtr;
    }


    private IntPtr CreateCudaStream()
    {
        var retPtr = PipelinePInvoke.CreateCudaStream();

        if (retPtr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateCudaStream] retPtr not init");
        }

        return retPtr;
    }

    private void CreateCppLogger(string fullLogPath)
    {
        if (string.IsNullOrWhiteSpace(fullLogPath))
            throw new ArgumentNullException(nameof(fullLogPath));

        PipelinePInvoke.InitLogger(new StringBuilder(fullLogPath));
    }

    private IntPtr CreateTrtEngin(YoloConfigs config)
    {
        if (string.IsNullOrWhiteSpace(config.EnginePath))
            throw new ArgumentNullException(nameof(config.EnginePath));

        if (config.DeviseId < 0)
            throw new ArgumentNullException(nameof(config.DeviseId));

        if (config.ConfThresh < 0 || config.ConfThresh > 1)
            throw new ArgumentNullException(nameof(config.ConfThresh));

        if (config.NmsThresh < 0 || config.NmsThresh > 1)
            throw new ArgumentNullException(nameof(config.NmsThresh));

        if (config.MaxNumOutputBbox < 0 || config.MaxNumOutputBbox > 1000)
            throw new ArgumentNullException(nameof(config.MaxNumOutputBbox));

        var configTrt = new TRTEngineConfig(config.EnginePath,
            config.DeviseId,
            config.ConfThresh,
            config.NmsThresh,
            config.MaxNumOutputBbox);

        var retTRTptr = PipelinePInvoke.CreateTRTEngine(ref configTrt, _cudaStream);

        if (retTRTptr == nint.Zero)
        {
            throw new Exception("[PipelineMlExtension:CreateTrtEngin] trtPtr not init");
        }

        return retTRTptr;
    }


    public void Dispose()
    {
        var arrDisposeObj = new[]
        {
            _pipeline,
            _gstDecoder,
            _trtEngine,
            _bufferFrameGpu,
            _bufferManager,
            _cudaStream,
            _encoder,
            _trackerManager
        };

        foreach (var currentObj in arrDisposeObj)
        {
            if (currentObj != nint.Zero)
            {
                PipelinePInvoke.Dispose(currentObj);
            }
        }
    }

    public DoInferenceRectDetectResult DoInferencePipeline()
    {
        unsafe
        {
            var pipelineOutputData = new PipelineOutputData();
            var ptr = nint.Zero;
            try
            {
                var result = PipelinePInvoke.DoInferencePipeline(_pipeline, ref pipelineOutputData);

                if (!result)
                    return new DoInferenceRectDetectResult { IsSuccess = false, RectDetects = [] };

                var retArray = new RectDetect[pipelineOutputData.RectanglesLen];
                ptr = (IntPtr)pipelineOutputData.Rectangles;
                var structSize = Marshal.SizeOf(typeof(RectDetectEngin));
                for (var i = 0; i < pipelineOutputData.RectanglesLen; i++)
                {
                    var currPtr = ptr + i * structSize;
                    var ptrToStructure = (RectDetectEngin)(Marshal.PtrToStructure(currPtr, typeof(RectDetectEngin))
                                                           ?? throw new InvalidOperationException(
                                                               "[DoInferencePipeline] Marshal.PtrToStructure return null"));

                    retArray[i] = RectDetect.RectDetectEnginToRectDetect(ptrToStructure);
                }

                return new DoInferenceRectDetectResult { IsSuccess = true, RectDetects = retArray };
            }
            catch (Exception e)
            {
                _logger.Error(e, "[DoInferencePipeline]");
                return new DoInferenceRectDetectResult { IsSuccess = false, RectDetects = [] };
            }
            finally
            {
                if (ptr != IntPtr.Zero)
                {
                    var result = PipelinePInvoke.DisposeArr(ptr);
                    if (!result)
                        throw new Exception("[DoInferencePipeline] Dispose return false");
                }
            }
        }
    }


    public DoInferenceImageResult GetCurrenImage()
    {
        var imageFrame = new ImageFrame();

        try
        {
            var result = PipelinePInvoke.GetCurrenImage(_pipeline, ref imageFrame);

            if (!result)
                return new DoInferenceImageResult() { IsSuccess = false, ImageInJpeg = [] };

            var retArray = new byte[imageFrame.ImageLen];

            Marshal.Copy(imageFrame.ImagesData, retArray, 0, retArray.Length);

            return new DoInferenceImageResult()
                { IsSuccess = true, ImageInJpeg = retArray, TimeStamp = imageFrame.TimeStamp };
        }
        catch (Exception e)
        {
            _logger.Error(e, "[GetCurrenImage]");
            return new DoInferenceImageResult() { IsSuccess = false, ImageInJpeg = [] };
        }
        finally
        {
            if (imageFrame.ImagesData != IntPtr.Zero)
            {
                var result = PipelinePInvoke.DisposeArr(imageFrame.ImagesData);
                if (!result)
                    throw new Exception("[GetCurrenImage] Dispose return false");
            }
        }
    }

    private static bool CreateWeight(ConverterConfig config)
    {
        if (!File.Exists(config.AssetsPath))
            return false;

        var pathFolderDestination = Path.GetDirectoryName(config.EnginePath);

        if (!Directory.Exists(pathFolderDestination))
            return false;

        if (File.Exists(config.EnginePath))
        {
            File.Delete(config.EnginePath);
        }

        var assetsPathChar = new StringBuilder(config.AssetsPath);
        var exportSaveChar = new StringBuilder(config.EnginePath);
        var inputLayer = config.InputLayer;
        var inputLayerCpp =
            new LayerSize(inputLayer.BatchSize, inputLayer.Channel, inputLayer.Width, inputLayer.Height);
        var idGpu = config.IdGpu;
        var setHalfModel = true;
        var res = PipelinePInvoke.ConverterNetworkWeight(assetsPathChar, exportSaveChar, ref inputLayerCpp, idGpu,
            setHalfModel);

        return res;
    }

    public bool StartPipelineGst(string connetctionString)
    {
        if (_gstDecoder != nint.Zero)
            PipelinePInvoke.Dispose(_gstDecoder);

        _gstDecoder = CreateGstDecoder();
        var connectionOnCameraPipliene = new StringBuilder(connetctionString);
        Log.Information("[StartPipelineGst] connectionOnCameraPipliene: {connectionOnCameraPipliene}",
            connectionOnCameraPipliene);

        var res = PipelinePInvoke.StartPipelineGst(_gstDecoder, connectionOnCameraPipliene);

        return res;
    }
}