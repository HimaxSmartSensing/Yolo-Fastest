# Yolo-Fastest person detection
This repository contains the person detection settings of the Yolo-Fastest model on the [COCO dataset](https://cocodataset.org/#home). And demonstrate how to train on the darknet platform and export the weight to the TensorFlow Lite model. This model is referenced from [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest).

## Prerequisites
- Build tools and environment settings for the darknet platform.
    - Please check [here](https://github.com/AlexeyAB/darknet#requirements-for-windows-linux-and-macos) to prepare the environment to build the darknet platform.
- Tensorflow model convert tool and the post-training quantization tool for this example. 
    - Please clone this [repository](https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set) to get the tools already set up for this example. This tool is  referenced from [david8862/keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set).
        ```bash
        git clone https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set
        ```
    - Please check [here](https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set#quick-start) to prepare the environment to run the convert tool.

## Dataset and Annotation files
- To get the COCO dataset can refer to [here](https://cocodataset.org/#download).
- Please check [here](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) to setting the all the objects that darknet needs.
- Change the data list file setting (`[train_coco.txt]` and `[test_coco.txt]`) at `ModelZoo/yolo-fastest-1.1_160_person/person.data`.
    ```
    classes= 1
    train  = [train_coco.txt]
    valid  = [test_coco.txt]
    names = person.names
    backup = model
    eval=coco
    ```
- Change the mapping of the image path to the label path [here](https://github.com/HimaxWiseEyePlus/Yolo-Fastest/blob/master/src/utils.c#L263).
    ```c++
    find_replace(input_path, "/images/train2017/", "/labels/train2017/", output_path);    // COCO
    find_replace(output_path, "/images/val2017/", "/labels/val2017/", output_path);        // COCO
    ```
- `annotation_file`: Image path and ground truth that convert tools need. The file formate can refer to [here](https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set#train). However, the ground truth label is not needed in the quantization stage or the prediction stage and can replace with `[train_coco.txt]` or `[test_coco.txt]` used by the darknet.
- `instances_json_file`: Used by [pycocotools](https://github.com/cocodataset/cocoapi) when calculating AP50 score. Can download it in the Annotations section [here](https://cocodataset.org/#download).

## Build
To set up other build settings of the darknet platform, please refer to [here](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make).
```bash
make
```

## Train Yolo-Fastest person detection model on COCO dataset

To Train Yolo-Fastest 160*160 resolution person detection model on COCO dataset:
```bash
cd ModelZoo/yolo-fastest-1.1_160_person/
../../darknet detector train person.data yolo-fastest-1.1_160_person.cfg
```

After the training progress, the weight file will be at `ModelZoo/yolo-fastest-1.1_160_person/model`. Our training results and other information for this model show in the table below.
Network| COCO 2017 Val person AP(0.5) |Resolution|FLOPS|Params|Weight size
:---:|:---:|:---:|:---:|:---:|:---:
[Yolo-Fastest-1.1_160_person](https://github.com/HimaxWiseEyePlus/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_160_person)|35.3 %|160*160|0.054BFlops|0.29M|1.15M|



## Convert weight to tensorflow
To convert the darknet weight to TensorFlow `.h5` file:
```bash
python keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
--config_path ModelZoo/yolo-fastest-1.1_160_person/yolo-fastest-1.1_160_person.cfg \
--output_path  ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.h5 \
--weights_path ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1_final.weights
...
Saved Keras model to ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.h5
Read 294252 of 294252.0 from Darknet weights.
```
After the convert progress, the `.h5` file will be at `ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.h5`.

## Post-training quantization and Evaluation 
To run the int8 quantization and output the `.tflite` model:
```bash
python keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py \
--keras_model_file ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.h5 \
--output_file  ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.tflite \
--annotation_file [annotation_file or train_coco.txt]
```

After the quantize progress, the `.tflite` file will be at `ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.tflite`.

To evaluate the tflite model with `pycocotools`:
```bash
python keras-YOLOv3-model-set/eval_yolo_fastest_160_1ch_tflite.py \
--model_path ModelZoo/yolo-fastest-1.1_160_person/model/yolo-fastest-1.1_160_person.tflite \
--anchors_path ModelZoo/yolo-fastest-1.1_160_person/anchors.txt \
--classes_path ModelZoo/yolo-fastest-1.1_160_person/person.names \
--json_name yolo-fastest-1.1_160_person.json \
--annotation_file [annotation_file or test_coco.txt]

Eval model: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [03:00<00:00, 27.66it/s]
The json result saved successfully.

python pycooc_person.py \
--res_path keras-YOLOv3-model-set/coco_results/yolo-fastest-1.1_160_person.json \
--instances_json_file [instances_json_file]

loading annotations into memory...
Done (t=0.41s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.97s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=13.08s).
Accumulating evaluation results...
DONE (t=0.97s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.140
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
```
The bounding box result will be at `keras-YOLOv3-model-set/coco_results/yolo-fastest-1.1_160_person.json`. The results of our evaluation of the quantized model show in the table below.
Network|COCO 2017 Val person AP(0.5)|
:---:|:---:
Yolo-Fastest-1.1_160_person int8|34.7 %|