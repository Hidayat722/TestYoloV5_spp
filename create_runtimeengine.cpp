#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include "yolo.h"
#include "yololayer.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
//#include "plugin_factory.h"

// +
extern "C"
bool NvDsInferCudaEngineGet(nvinfer1::IBuilder *builder,
        NvDsInferContextInitParams *initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{
    //std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::cout<<"Creating Engine file"<<std::endl;
    std::string yoloType = "yolov5";
    IHostMemory* modelStream{nullptr};
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p("yolov3-spp.engine");
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return true;
    
}

bool NvDsInferPluginFactoryRuntimeGet (nvinfer1::IPluginFactory *& pluginFactory) {
	std::cerr << "called" << std::endl;
	auto pf = new YoloLayerPlugin();
	//pluginFactory = new PluginFactory();
	std::cout << "pluginFactory: " << pf << std::endl;
	//pluginFactory = pf;
	return pf != nullptr;
}

