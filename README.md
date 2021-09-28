# Yolo-Fastest person detection
This repository contains the person detection settings of the Yolo-Fastest model on the [COCO dataset](https://cocodataset.org/#home). And demonstrate how to train on the darknet platform and export the weight to the TensorFlow Lite model. This model is referenced from [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest).

## Prerequisites
- Build tools and environment settings for the darknet platform.
    - Please check [here](https://github.com/AlexeyAB/darknet#requirements-for-windows-linux-and-macos) to prepare the environment to build the darknet platform.
- Tensorflow model convert tool and the post-training quantization tool for this example. 
    - Please check [here](https://github.com/HimaxWiseEyePlus/keras-YOLOv3-model-set#quick-start) to prepare the environment to run the convert tool. This tool is referenced from [david8862/keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set).

## Dataset and Annotation files
- To get the COCO 2017 dataset can refer to [COCO 2017 train images dataset](http://images.cocodataset.org/zips/train2017.zip) and [COCO 2017 val images dataset](http://images.cocodataset.org/zips/val2017.zip).
- `instances_json_file`: Used by [pycocotools](https://github.com/cocodataset/cocoapi) when calculating AP50 score. The COCO 2017 annotations can download from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
- Please check [here](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) to setting the all the objects that darknet needs. In this example, some settings have been prepared in advance, only need to prepare the following items:
    - Create file `[train_coco.txt]` and `[test_coco.txt]`, with the file path of your training images and validation images. For example:
        ```
        /images/train2017/000000337711.jpg
        /images/train2017/000000300758.jpg
        /images/train2017/000000494434.jpg
        ...
        ```
    - Create a `.txt` file for each `.jpg` image file in the same directory and with the same name to the image file. The file contains the object number and the object coordinates on this image, for each object in a new line:
        ```
        <object-class> <x_center> <y_center> <width> <height>
        ```
        Where:
        - `<object-class>` - integer object number from 0 to (classes-1), in this example only use 0 for person label.
        - `<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
        - Example:
            ```
            0 0.686445 0.53196 0.0828906 0.323967
            0 0.612484 0.446197 0.023625 0.0838967
            ```
    We have a modified version of the  [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) in this repository.(Modify to be used for the person detection task) Can use this tool to convert the COCO `instances_json_file` JSON into darknet format:
    ```bash 
    # Python 3.8 or later
    pip install -r ./JSON2YOLO/requirements.txt
    python JSON2YOLO/general_json2yolo.py \
    --instances_path [instances_json_file directory] \
    --train_path [training dataset directory] \
    --val_path [val dataset directory]

    Annotations annotations/instances_train2017.json: 100%|█████████████████████████████████████████████████| 860001/860001 [43:02<00:00, 333.05it/s]
    Annotations annotations/instances_val2017.json: 100%|█████████████████████████████████████████████████████| 36781/36781 [01:38<00:00, 372.40it/s]
    Labels path: new_dir/labels
    Data list files: new_dir/train_coco.txt, new_dir/test_coco.txt
    ```

- Change the data list file path setting (`[train_coco.txt]` and `[test_coco.txt]`) at `ModelZoo/yolo-fastest-1.1_160_person/person.data`.
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

|Network| COCO 2017 Val person AP(0.5) |Resolution|FLOPS|Params|Weight size|
|:---:|:---:|:---:|:---:|:---:|:---:|
|[Yolo-Fastest-1.1_160_person](https://github.com/HimaxWiseEyePlus/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_160_person)|35.3 %|160*160|0.054BFlops|0.29M|1.15M|



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
```
```bash
python pycooc_person.py \
--res_path keras-YOLOv3-model-set/coco_results/yolo-fastest-1.1_160_person.json \
--instances_json_file [val_instances_json_file]

loading annotations into memory...
Done (t=0.47s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.98s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=14.29s).
Accumulating evaluation results...
DONE (t=1.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.140
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
```
The bounding box result will be at `keras-YOLOv3-model-set/coco_results/yolo-fastest-1.1_160_person.json`. The results of our evaluation of the quantized model show in the table below.

|Network|COCO 2017 Val person AP(0.5)|
:---:|:---:|
|[Yolo-Fastest-1.1_160_person int8](https://github.com/HimaxWiseEyePlus/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_160_person/yolo-fastest-1.1_160_person.tflite)|34.8 %|


## Himax pretrained model
For the tinyML model, if the training dataset is collected from the hardware used by the deployment target and the collecting scenarios are consistent with the usage scenarios. The model trained with these data can usually have better accuracy when running on the deployment target.
To this end, we collected approximately 180,000 pictures of himax office scenes using himax cameras. Training on this example model and take 20% of the data for validation. The `.tflite` file and the validation results of this model show in the following table:
|Network|Validation AP(0.5)|
:---:|:---:|
|[Yolo-Fastest-1.1_160_person_himax int8](https://github.com/HimaxWiseEyePlus/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_160_person/yolo-fastest-1_1_160_person_himax.tflite)|89.2 %|

## Thanks
- https://github.com/AlexeyAB/darknet
- https://github.com/dog-qiuqiu/Yolo-Fastest
- https://github.com/david8862/keras-YOLOv3-model-set
- https://github.com/ultralytics/JSON2YOLO
