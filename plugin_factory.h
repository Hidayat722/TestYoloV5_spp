#ifndef MY_PLUGIN_FACTORY_H
#define MY_PLUGIN_FACTORY_H
#include <NvInfer.h>
#include <iostream>
#include "yololayer.h"

namespace nvinfer1 {
class PluginFactory : public IPluginFactory {
    public:
        IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
};
/*
class YoloCustomLayerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    YoloCustomLayerPluginCreator () {}
    ~YoloCustomLayerPluginCreator () {}

    const char* getPluginName () const override { return "YOLOCUSTOM"; }
    const char* getPluginVersion () const override { return "1"; }

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cerr<< "YoloCustomLayerPluginCreator::getFieldNames is not implemented" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2* createPlugin (
        const char* name, const nvinfer1::PluginFieldCollection* fc) override
    {
        std::cerr<< "YoloCustomLayerPluginCreator::getFieldNames is not implemented.\n";
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin (
        const char* name, const void* serialData, size_t serialLength) override
    {
        std::cout << "Deserialize yoloLayer plugin: " << name << std::endl;
        return new YoloLayerPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override {
        m_Namespace = libNamespace;
    }
    const char* getPluginNamespace() const override {
        return m_Namespace.c_str();
    }

private:
    std::string m_Namespace {""};
};
*/
}
#endif
