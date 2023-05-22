import os
import cv2
import sys
import onnx
import numpy as np

import onnxruntime as ort
from PIL import Image

# Normalize an image.
def normalize(im, mean, std):

    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im

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
    color_map = [0, 0, 0,
                255, 255, 255]
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')

    #颜色填充
    pred_mask.putpalette(color_map)

   
    return pred_mask


def migraphx_seg(model_path,image):
    import migraphx


    print(image.shape)
    maxInput={"x":[1,3,605,700]}

    model = migraphx.parse_onnx(model_path,map_input_dims=maxInput)

    inputName=model.get_parameter_names()[0]
    inputShape=model.get_parameter_shapes()[inputName].lens()
    print("inputName:{0} \ninputShape:{1}".format(inputName,inputShape))

    model.compile(t=migraphx.get_target("gpu"),device_id=0)

    inputShape=[1,3,605,700]

    inputShapeMap={inputName:inputShape}

    #设置输入
    migraphx.reshape2(model, inputShapeMap)


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
    print(scores)    
    print("ort result.shape:",scores.shape)

    return scores

def main():
    #加载模型
    model_path="unet-stare.onnx"

    #读取图像
    image_CV = cv2.imread('im0001.ppm')

    #图像预处理
    image=Imagepreprocess(image_CV)

    # #ort推理
    # scores=ort_seg_dcu(model_path,image)
                       
    #migraphx 推理
    scores=migraphx_seg(model_path,image)
    
    #生成MASK图像
    pred_mask = get_color_mask(scores)
    
    # 保存图像分割结果
    pred_mask.save('result/Mi0001.png')





if __name__ == '__main__':
    main()