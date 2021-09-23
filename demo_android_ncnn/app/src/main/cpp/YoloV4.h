#ifndef YOLOV4_H
#define YOLOV4_H

#include "net.h"
#include "YoloV5.h"


class YoloV4 {
public:
    YoloV4(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);

    ~YoloV4();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float threshold, float nms_threshold);
    std::vector<std::string> labels{"gatorade", "pepsi"};
private:
    static std::vector<BoxInfo>
    decode_infer(ncnn::Mat &data, const yolocv::YoloSize &frame_size, int net_size, int num_classes, float threshold);

//    static void nms(std::vector<BoxInfo>& result,float nms_threshold);
    ncnn::Net *Net;
    int input_size = 416;
    int num_class = 2;
public:
    static YoloV4 *detector;
    static bool hasGPU;
};


#endif //YOLOV4_H
