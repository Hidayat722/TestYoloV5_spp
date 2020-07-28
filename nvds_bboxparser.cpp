// +
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "common.h"

#include <opencv2/core.hpp>

#define NMS_THRESH 0.5
#define CONF_THRESH 0.4
#define BATCH_SIZE 1

static const int INPUT_H = 608;
static const int INPUT_W =608;
static const int image_width = 1280;
static const int image_height = 720;
// -


extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
     std::cout<<"parser function called ----"<<std::endl;
     //const NvDsInferLayerInfo &layer = outputLayersInfo[0]; // num_boxes x (4 + num_classes)
     std::vector<Yolo::Detection>res;
     
    for(int i=0; i<outputLayersInfo.size(); i++) {
	    std::cout << outputLayersInfo[i].layerName << std::endl;
    }
    //std::cout << cv::Mat(600,600,CV_32FC1,outputLayersInfo[0].buffer) << std::endl;

    nms(res, (float*)(outputLayersInfo[0].buffer), CONF_THRESH, NMS_THRESH);
    std::cout<<"Nms done sucessfully----"<<std::endl;
    /*
    for (unsigned int i = 0; i < res.size(); i++ ){
        NvDsInferParseObjectInfo b;
        cv::Rect r = get_rect(image_width, image_height, res[i].bbox);
        b.top= (unsigned int) r.y;
        b.left= (unsigned int) r.x;
        b.width= (unsigned int)r.width;
        b.height= (unsigned int)r.height;
        b.detectionConfidence= res[i].conf;
        b.classId= (unsigned int)res[i].class_id;
        objectList.push_back(b);
        
    }
    */
    
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;
	    oinfo.left = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
	    objectList.push_back(oinfo);
        
    }
    
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
