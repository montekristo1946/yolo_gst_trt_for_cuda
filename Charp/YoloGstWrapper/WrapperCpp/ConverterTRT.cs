using System.Security.Cryptography;
using System.Text;
using WrapperCpp.Configs;
using WrapperCpp.InfrastructureCPP;
using WrapperCpp.InfrastructureCPP.PInvokeDto;

namespace WrapperCpp;

public static class ConverterTRT
{
    public static bool RunConverter(ConverterConfig converterConfig, string fullLogPath)
    {
        if (converterConfig is null)
            throw new ArgumentNullException(nameof(converterConfig));

        var resCheckSignEngin = CheckNets(converterConfig.EnginePath, converterConfig.AssetsPath);

        if (resCheckSignEngin)
        {
            return true;
        }

        CreateCppLogger(fullLogPath);

        var resCreateWeight = CreateWeight(converterConfig);

        if (!resCreateWeight)
        {
            return false;
        }

        var resCreateHash = WriteHash(converterConfig.AssetsPath, converterConfig.EnginePath);

        return resCreateHash;
    }

    private static void CreateCppLogger(string fullLogPath)
    {
        var logPath = new StringBuilder(fullLogPath);

        PipelinePInvoke.InitLogger(logPath);
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
        var inputLayer     = config.InputLayer;
        var inputLayerCpp  = new LayerSize(inputLayer.BatchSize, inputLayer.Channel, inputLayer.Width, inputLayer.Height);
        var idGpu          = config.IdGpu;
        var setHalfModel   = true;
        var res            = PipelinePInvoke.ConverterNetworkWeight(assetsPathChar, exportSaveChar, ref inputLayerCpp, idGpu, setHalfModel);

        return res;
    }

    private static bool WriteHash(string assetsPath, string enginePath)
    {
        if (string.IsNullOrEmpty(assetsPath)|| string.IsNullOrEmpty(enginePath))
            return false;

        if (!File.Exists(assetsPath))
            return false;

        var hashWeight = GetMd5Hash(assetsPath);

        if (string.IsNullOrEmpty(hashWeight))
            return false;

        var pathFolderDestination = Path.GetDirectoryName(enginePath);

        if (!Directory.Exists(pathFolderDestination))
            return false;

        var weightPathHash = Path.Combine(pathFolderDestination, $"{Path.GetFileName(assetsPath)}.hash");

        if (File.Exists(weightPathHash))
        {
            File.Delete(weightPathHash);
        }

        File.WriteAllText(weightPathHash, hashWeight);

        return true;
    }

    private static bool CheckNets(string enginePath, string assetsPath)
    {
        if (string.IsNullOrEmpty(enginePath) || string.IsNullOrEmpty(assetsPath))
            return false;

        var pathFolderDestination = Path.GetDirectoryName(enginePath);

        if (pathFolderDestination is null)
            return false;

        var weightPathHash = Path.Combine(pathFolderDestination, $"{Path.GetFileName(assetsPath)}.hash");

        if (!File.Exists(enginePath) || !File.Exists(weightPathHash) || !File.Exists(assetsPath))
            return false;

        var hashWeightSrc = GetMd5Hash(assetsPath);

        if (string.IsNullOrEmpty(hashWeightSrc))
            return false;

        var hashInFiles = File.ReadAllText(weightPathHash);

        if (hashWeightSrc != hashInFiles)
            return false;

        return true;
    }

    private static string GetMd5Hash(string path)
    {
        if (string.IsNullOrEmpty(path))
            return string.Empty;

        using var stream   = File.OpenRead(path);
        using var provider = MD5.Create();

        var checkSum = provider.ComputeHash(stream);

        return BitConverter.ToString(checkSum)
            .Replace("-", string.Empty)
            .ToLower();
    }
}
