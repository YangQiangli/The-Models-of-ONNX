import os
import cv2
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort



# 生成colormap
def get_color_map_list(num_classes, custom_color=None):


    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map

#  正则化.
def normalize(im, mean, std):

    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im
#   图像处理
def Imagepreprocess(image):


    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)

    mean_ = np.array(mean)[np.newaxis, np.newaxis, :]
    std_ = np.array(std)[np.newaxis, np.newaxis, :]
    img = normalize(image, mean_, std_)

    img = np.expand_dims(img.transpose(2, 0, 1), 0)
    return img


def get_color_mask(pred, color_map=None):


    pred = np.squeeze(pred, 0)
    pred = pred.astype('uint8')
    
    #用于生成mask的颜色库
    color_map = get_color_map_list(19)
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')

    #颜色填充
    pred_mask.putpalette(color_map)

   
    return pred_mask


def migraphx_seg(model_path,image):
    import migraphx


    print(image.shape)

    inputName=model.get_parameter_names()[0]
    maxInput={inputName:[1,3,1024,2048]}
    model = migraphx.parse_onnx(model_path,map_input_dims=maxInput)

    inputName=model.get_parameter_names()[0]
    inputShape=model.get_parameter_shapes()[inputName].lens()
    print("inputName:{0} \ninputShape:{1}".format(inputName,inputShape))

    model.compile(t=migraphx.get_target("gpu"),device_id=0)

    # inputShape=[1,3,1024,2048]

    # inputShapeMap={inputName:inputShape}

    # #设置输入
    # migraphx.reshape2(model, inputShapeMap)


    results = model.run({inputName: migraphx.argument(image)})
    scores=np.array(results[0])
    print("migraphx result.shape:",scores.shape)

    return scores


def ort_seg_dcu(model_path,image):
    
    #创建sess_options
    sess_options = ort.SessionOptions()

    #设置图优化
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    #是否开启profiling
    sess_options.enable_profiling = False
    dcu_session = ort.InferenceSession(model_path,sess_options,providers=['ROCMExecutionProvider'],)
    input_name=dcu_session.get_inputs()[0].name

    results = dcu_session.run(None, input_feed={input_name:image })
    scores=np.array(results[0])
    print("ort result.shape:",scores.shape)

    return scores

def ort_seg_cpu(model_path,image):
    cpu_session=ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name=cpu_session.get_inputs()[0].name
    results = cpu_session.run(None, input_feed={input_name:image })
    scores=np.array(results[0])
 
    print("ort result.shape:",scores.shape)

    return scores

def main():
    #加载模型
    model_path="model/deeplabv3_r101.onnx"

    #读取图像
    image_CV = cv2.imread('testimg.png')

    #图像预处理
    image=Imagepreprocess(image_CV)

    # #ort_cpu推理
    scores=ort_seg_cpu(model_path,image)

    # #ort_dcu推理  //
    # scores=ort_seg_dcu(model_path,image)
                       
    # #migraphx 推理
    # scores=migraphx_seg(model_path,image)
    
    #生成MASK图像
    pred_mask = get_color_mask(scores)
    
    # 保存图像分割结果
    pred_mask.save('result/testimg_cpu.png')





if __name__ == '__main__':
    main()