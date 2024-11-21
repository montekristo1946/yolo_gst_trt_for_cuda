namespace WrapperCpp.Configs;

public class ConverterConfig
{
    /// <summary>
    ///     Путь модели сети выходной
    /// </summary>
    public string EnginePath { get; set; } = string.Empty;

    /// <summary>
    ///     Путь до Asset, входной
    /// </summary>
    public string AssetsPath { get; set; } = string.Empty;

    /// <summary>
    ///     Конфигурация входного слоя
    /// </summary>
    public LayerSizeConfig InputLayer { get; set; } = new LayerSizeConfig();

    /// <summary>
    ///     ID видеоУскорителя
    /// </summary>
    public int IdGpu { get; set; } = 0;
}
