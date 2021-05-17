# This script takes a youtube video stream and saves an image every 5 seconds in a cache directory
import time
import streamlink as sl
import cv2
import requests
from utils import load_labels, get_current_time, draw_bbox, image_stats, update_entity
    
    
# Video url
yturl = 'https://www.youtube.com/watch?v=AdUw5RdyZxI'
quality = '720p'

streamlist = sl.streams(yturl)
url = streamlist[quality].url

# Start capture
capture = cv2.VideoCapture(url)
if not capture.isOpened:
    print('Unable to open: ' + url)
    exit(-1)

# Loop over frames
counter = 0
start_time= time.time()
while True:
    ret, frame = capture.read()
    end_time = time.time()
    print(end_time-start_time)
    if (end_time-start_time) > 5:
        name = str(counter) + '.jpg'
        cv2.imwrite('./cache/' + name , frame)
        counter = counter + 1
        start_time= time.time()
