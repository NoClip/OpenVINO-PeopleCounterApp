# Project Write-Up
The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count

## Explaining Custom Layers

No custom layers used.


## Comparing Model Performance

I couldn't able to run the preconverted tensorflow model, I searched and quoted the speed data from the internet.

| Metric        | Pre-Conversion|   Post-Conversion   |
| ------------- | ------------- | ------------------- |
| Size          | 69.6883 MB    | 67.3858 MB (bin+xml)|
| Speed         |      27 ms    |      61 ms          |


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

- Model Accuracy: It's the most necessary steps before implemented this project in the edge. Because low accuracy model will provide bad result (as label will not be detected or false label will be detected). For that reason, high accuracy model is must necessary in the edge application.

- Camera Focal Length: The longer the focal length the more people detected, but I it may affects the accuracy. Also image size result more accuracy, but might affects the processing if its too large.


## Model Research

In investigating potential people counter models, I tried each of the following models:

- Model 1: [faster_rcnn_inception_v2_coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments frozen_inference_graph.pb, pipeline.config and faster_rcnn_support.json files.
  - The model was insufficient for the app because the inference time was too much although it correctly detected people.
  - I thought to skip frames to speed up the inference time, but it wouldn't be accurate.


- Model 2: [ssd_mobilenet_v2_coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments frozen_inference_graph.pb, pipeline.config and ssd_support.json files.
  - The model was sufficient for the app because the inference time was fast, but it detected the same person many times.
  - I tried to improve the model for the app by checking the person if he left the scene then I count, that helped not detecting the same person many times.
