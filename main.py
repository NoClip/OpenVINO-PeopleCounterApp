"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

COCO_CLASSES = ['__background__','person','bicycle','car','motorcycle','airplane',
           'bus','train','truck','boat','traffic light','fire hydrant','stop sign',
           'parking meter','bench','bird','cat','dog','horse','sheep','cow']

log.basicConfig(level=log.INFO)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    last_count = 0
    total_count = 0
    start_time = 0
    duration = 0
    frame_counter = 0
    
    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    # Create a flag for single images
    single_image_mode = False

    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    # initial width
    initial_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # id =3
    # initial height
    initial_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # id = 4
    
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    frames_skip = 20
    current_counts_array = np.zeros(frames_count)

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        frame_counter += 1

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = preprocessImage(frame, n, c, h, w)

        ### TODO: Start asynchronous inference for specified request ###
        infer_start_time = time.time()
        infer_network.exec_net(image)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            infer_time_diff = time.time() - infer_start_time

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            frame, current_count = draw_boxes(frame, result, args, initial_width, initial_height)

            current_counts_array[frame_counter] = current_count

            inf_time_message = "Inference time: {:.3f}ms".format(infer_time_diff * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
          
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            if frame_counter > frames_skip:
                if 1 in current_counts_array[frame_counter - frames_skip : frame_counter]:
                    current_count = 1

                # When new person enters the video
                if current_count > last_count:
                    start_time = time.time()
                    total_count += current_count - last_count
                    client.publish("person", json.dumps({"total": total_count}))

                # Person duration in the video is calculated
                if current_count < last_count:
                    duration = int(time.time() - start_time)
                    # Publish messages to the MQTT server
                    client.publish("person/duration", json.dumps({"duration": duration}))

                last_count = current_count

            client.publish("person", json.dumps({"count": current_count}))

            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get("BLUE")
    
    current_count = 0
    
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if box[1] == 1 and conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), out_color, 1)
            current_count = current_count + 1
            
    return frame, current_count

def preprocessImage(frame, n, c, h, w):
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
