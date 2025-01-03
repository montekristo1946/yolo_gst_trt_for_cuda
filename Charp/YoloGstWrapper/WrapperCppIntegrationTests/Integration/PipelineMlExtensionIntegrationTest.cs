using System.Diagnostics;
using WrapperCpp;
using WrapperCpp.Configs;
using WrapperCpp.InfrastructureCPP;

namespace WrapperCppTests.Integration;

public class PipelineMlExtensionIntegrationTest
{
    private TrackerConfig CreateTrackerConfig()
    {
        return new TrackerConfig()
        {
            FrameRate = 30,
            TrackBuffer = 30,
            TrackThresh = 0.2F,
            HighThresh = 0.5F,
            MathThresh = 0.7F,
            MaxTrack = 4
        };
    }
    
    private static YoloConfigs CreateYoloConfigs()
    {
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
            // ConnetctionString = connection,
            PathLogFile = "./Logs/PipelineMl_FullPass.txt"
        };
        return config;
    }
    private static string CreateConnection()
    {
        var connection = "filesrc location=/mnt/Disk_D/Document/Teplovisors/Dataset/010/09.09.2024_002.avi " +
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
        return connection;
    }
    
    public static Polygon[] CreateMoqPolygon()
    {
        return
        [
            new Polygon()
            {
                Id = 1,
                Points =
                [
                    new PointWrapCpp() { Id = 0, X = 0.01F, Y = 0.01F },
                    new PointWrapCpp() { Id = 1, X = 0.99F, Y = 0.01F },
                    new PointWrapCpp() { Id = 2, X = 0.99F, Y = 0.49F },
                    new PointWrapCpp() { Id = 3, X = 0.01F, Y = 0.49F },
                ]
            },
            new Polygon()
            {
                Id = 2,
                Points =
                [
                    new PointWrapCpp() { Id = 0, X = 0.01F, Y = 0.51F },
                    new PointWrapCpp() { Id = 1, X = 0.99F, Y = 0.51F },
                    new PointWrapCpp() { Id = 2, X = 0.99F, Y = 0.99F },
                    new PointWrapCpp() { Id = 3, X = 0.01F, Y = 0.99F },
                ]
            },
        ];
    }

    public void FullPass()
    {
        var connectionUrl = CreateConnection();

        var yoloConfigs = CreateYoloConfigs();
        var trackerConfig = CreateTrackerConfig();
        var configPolygons = CreateMoqPolygon();
        
        var pipelineMl = new PipelineMlExtension(yoloConfigs,trackerConfig,configPolygons);
        pipelineMl.StartPipelineGst(connectionUrl);
        

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