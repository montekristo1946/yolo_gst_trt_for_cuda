using Serilog;
using Serilog.Events;
using WrapperCppTests.Integration;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Is(LogEventLevel.Debug)
    .WriteTo.Console()
    .CreateLogger();

ILogger _logger = Log.ForContext<Program>();

_logger.Debug("RUNNING  WrapperCppUnitTests");

// new PipelineMlExtensionIntegrationTest().CreateTRTWeight();
new PipelineMlExtensionIntegrationTest().FullPass();
// new PipelineMlExtensionIntegrationTest().TestMemoryLeek();