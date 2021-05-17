# This script reads images from the cache directory, sends the image to a bentoml service to detect objects and visualises the response
import time
import cv2
import requests
from utils import load_labels, get_current_time, draw_bbox, image_stats, update_entity
import os

# Loop over images
while True:
    
    image_paths = [os.path.join('./cache', path) for path in os.listdir('./cache') ]
    image_paths.sort(key=os.path.getmtime)
    frame = cv2.imread(image_paths[0])   
    # Save image to cache
    files = {'image': open(image_paths[0], 'rb')}
    
    # Send image to bentoml service for prediction
    response = requests.post("http://0.0.0.0:5000/predict", files = files)
    detections = response.json()
    
    # Transform predictions into statistics
    stats, p_bboxes = image_stats(frame, detections)
    print(stats)
    
    # Send statistics to the context broker
    # response = update_entity(stats)
    # print(response)
    
    # Visualise detections
    frame = draw_bbox(frame, p_bboxes, classes=load_labels("coco.names"))
    cv2.imshow('Frame', frame)
    time.sleep(5)
    keyboard = cv2.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
    os.remove(image_paths[0])
