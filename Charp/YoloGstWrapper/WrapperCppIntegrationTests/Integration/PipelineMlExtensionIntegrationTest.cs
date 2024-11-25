using System.Diagnostics;
using WrapperCpp;
using WrapperCpp.Configs;
using WrapperCpp.InfrastructureCPP;

namespace WrapperCppTests.Integration;

public class PipelineMlExtensionIntegrationTest
{
    public void FullPass()
    {
        var connection = "filesrc location=/mnt/Disk_D/Document/Teplovisors/Dataset/010/11.09.2024_001.avi " +
                         "! avidemux " +
                         "! nvv4l2decoder " +
                         "! nvvideoconvert nvbuf-memory-type=3 " +
                         "! video/x-raw(memory:NVMM)" +
                         "! appsink name=mysink sync=true";

        // var connection = "rtspsrc location=rtsp://login:password@192.168.1.15:554 latency=1000 " +
        //                  "! rtph264depay " +
        //                  "! nvv4l2decoder " +
        //                  "! nvvideoconvert nvbuf-memory-type=3 " +
        //                  "! video/x-raw(memory:NVMM) " +
        //                  "! appsink name=mysink sync=true";

        var config = new YoloConfigs()
        {
            EnginePath = "/mnt/Disk_C/git/yolo_gst_trt_for_cuda/CPP/weight/model_001.engine",
            DeviseId = 0,
            ConfThresh = 0.1F,
            NmsThresh = 0.6F,
            MaxNumOutputBbox = 1000,
            WidthImgMl = 640,
            HeightImgMl = 640,
            CountImgToBackground = 25,
            ConnetctionString = connection,
            PathLogFile = "./Logs/PipelineMl_FullPass.txt"
        };

        var pipelineMl = new PipelineMlExtension(config);

        var stopwatch = new Stopwatch();
        var interation = 10;
        while (interation > 0)
        {
            Thread.Sleep(1);
            stopwatch.Restart();
                var resDoInferencePipeline = pipelineMl.DoInferencePipeline();

            if (!resDoInferencePipeline.IsSuccess)
                continue;
            interation -= 1;
            var resGetCurrenImage = pipelineMl.GetCurrenImage();


            stopwatch.Stop();

            if (resGetCurrenImage.IsSuccess)
            {
                var time = resGetCurrenImage.TimeStamp;
                var filePath = $"/mnt/Disk_D/TMP/20.11.2024/{time/1000000}.jpg";
                
                File.WriteAllBytes(filePath, resGetCurrenImage.ImageInJpeg);
            }

            Console.WriteLine($"resDoInference: {resDoInferencePipeline.IsSuccess} " +
                              $"rectangles:{resDoInferencePipeline.RectDetects.Length} " +
                              $"TimeStamprectangles:{resDoInferencePipeline.RectDetects.FirstOrDefault()?.TimeStamp} " +
                              $"ElapsedM:{stopwatch.ElapsedMilliseconds} " +
                              $"Length:{resGetCurrenImage.ImageInJpeg.Length} " +
                              $"TimeStampIMg:{resGetCurrenImage.TimeStamp}");
        }

        pipelineMl.Dispose();
    }


    public void TestMemoryLeek()
    {
        for (int i = 0; i < 1000; i++)
        {
            Console.WriteLine($"-----{i}-----");
            FullPass();
        }
    }

    public void CreateTRTWeight()
    {
        var config = new ConverterConfig()
        {
            EnginePath = "/mnt/Disk_C/git/yolo_gst_trt_for_cuda/CPP/weight/model_001.engine",
            AssetsPath = "/mnt/Disk_C/git/yolo_gst_trt_for_cuda/CPP/weight/model_001.onnx",
            IdGpu = 0,
            InputLayer = new LayerSizeConfig()
            {
                BatchSize = 1,
                Channel = 3,
                Height = 640,
                Width = 640
            },
        };
        var res = ConverterTRT.RunConverter(config, "./Logs/ConverterTRT.txt");
        
        if(!res)
            throw new Exception("Create TRT weight failed");
    }
}