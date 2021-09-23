//
// Create by RangiLyu
// 2020 / 10 / 2
//

#ifndef NANODET_H
#define NANODET_H

#include "net.h"
#include "YoloV5.h"

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;


class NanoDet{
public:
    NanoDet(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);

    ~NanoDet();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float score_threshold, float nms_threshold);
    std::vector<std::string> labels{"gatorade", "pepsi"};
private:
    void preprocess(JNIEnv *env, jobject image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>>& results, float width_ratio, float height_ratio);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, float width_ratio, float height_ratio);

    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

    ncnn::Net *Net;
    int input_size = 416;
    int num_class = 2;
    int reg_max = 7;
    std::vector<HeadInfo> heads_info{
        // cls_pred|dis_pred|stride
        {"cls_pred_stride_8", "dis_pred_stride_8", 8},
        {"cls_pred_stride_16", "dis_pred_stride_16", 16},
        {"cls_pred_stride_32", "dis_pred_stride_32", 32},
    };

public:
    static NanoDet *detector;
    static bool hasGPU;
};


#endif //NANODET_H
