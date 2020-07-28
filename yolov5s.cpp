// +
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.h"
#include "yolo.h"
#include <map>

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

// +
// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("yolov5s.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.18.weight"], weightMap["model.18.bias"]);

    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.19");
    ITensor* inputTensors20[] = {conv19->getOutput(0), conv14->getOutput(0)};
    auto cat20 = network->addConcatenation(inputTensors20, 2);
    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.21");
    IConvolutionLayer* conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.22.weight"], weightMap["model.22.bias"]);

    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), 256, 3, 2, 1, "model.23");
    ITensor* inputTensors24[] = {conv23->getOutput(0), conv10->getOutput(0)};
    auto cat24 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.25");
    IConvolutionLayer* conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.26.weight"], weightMap["model.26.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

std::map <std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}
            
            void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

