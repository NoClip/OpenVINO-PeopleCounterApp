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

# First 20 COCO Classes
CLASSES = ['__background__','person','bicycle','car','motorcycle','airplane',
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
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
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

    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0

    
    no_box = 0
    prev_box_x = 0
    start_time = 0
    duration = 0
    
    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 
                                          1, 1, cur_request_id, 
                                          args.cpu_extension)[1]

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
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        #net_input = {'image_tensor': image,'image_info': image.shape[1:]}
        
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, image)
        
        
        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id)
            if args.perf_counts:
                perf_count = infer_network.performance_counter(cur_request_id)
                performance_counts(perf_count)
            
            ### TODO: Extract any desired stats from the results ###
            frame, current_count, top_left, bottom_right = draw_boxes(frame, result, args, initial_width, initial_height)
            box_w = frame.shape[1]
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            inf_time_message = "Averagetime: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                no_box = 0
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                if no_box <= 20:
                    current_count = last_count
                    no_box += 1
                elif prev_box_x < box_w - 200:
                    current_count = last_count
                    no_box = 0
                else:
                    duration = int(time.time() - start_time)
                    # Publish messages to the MQTT server
                    client.publish("person/duration", json.dumps({"duration": duration}))

            if top_left != None and bottom_right != None:
                prev_box_x = int((top_left[0] + bottom_right[0]) / 2)

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count       
            
            
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
    p1 = None
    p2 = None
    
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if box[1] == 1 and conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            p1 = (xmin, ymin)
            p2 = (xmax, ymax)
            cv2.rectangle(frame, p1, p2, out_color, 1)
            current_count = current_count + 1
    return frame, current_count, p1, p2


def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


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
