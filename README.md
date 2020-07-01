# Project Write-Up

## Explaining Custom Layers

No custom layers used.


## Comparing Model Performance

I couldn't able to run the preconverted tensorflow model, I searched and quoted the speed data from the internet.

| Metric        | Pre-Conversion|   Post-Conversion   |
| ------------- | ------------- | ------------------- |
| Size          | 69.6883 MB    | 67.3858 MB (bin+xml)|
| Speed         |      27 ms    |      61 ms          |

Used parameters:
`python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

## Assess Model Use Cases
* Social distance warnings.
* Count people in certain situations like:
    * public transportation, super market, conferences' for statistics
    * classroom for checking insructors/students presents
* Validate who enters a buildings (home, office) for security.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting: Lighting should setup properly to get a better video quality, that will get a better results.

- Model Accuracy: It doesn't detect all people in all posses.

- Camera Focal Length: The longer the focal length the more people detected, but I it may affects the accuracy. Also image size/resolution results more accuracy, but might affects the processing if its too large.


## Model Research

In investigating potential people counter models, I tried each of the following models:

- Model 1: [faster_rcnn_inception_v2_coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments (where $1 is model path):
    `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model $1/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config $1/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --input_shape "[1,600,1024,3]" -o /home/workspace/models/tfrcnn --data_type FP16`
  - The model was insufficient for the app because the inference time was too much although it correctly detected people.
  - I thought to skip frames to speed up the inference time, but it wouldn't be accurate.

- Model 2: [ssd_mobilenet_v2_coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments (where $1 is model path):
  `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model $1/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config $1/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json -o /home/workspace/models`
  - The model was sufficient for the app because the inference time was fast, but it detected the same person many times.
  - I tried to improve the model by skipping frames (not an accurate way), it kind of worked but it counts a person twice.
