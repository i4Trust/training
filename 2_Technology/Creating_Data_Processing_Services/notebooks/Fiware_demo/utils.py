import cv2
import datetime
import random
import numpy as np
import requests
import colorsys
import json

from collections import Counter
from sklearn.cluster import AgglomerativeClustering


# Read class names from obj.names
def load_labels(path):
    classes = []
    with open(path, "r") as f:
        classes = [cname.strip() for cname in f.readlines()]
    return classes
    
# Get time in ISO 8601 format
def get_current_time():
    return datetime.datetime.now().isoformat()[:-3]+'Z'

# Draw detections on image
def draw_bbox(image, bboxes, classes=load_labels("coco.names"), show_label=False):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    color_num = 25
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / color_num, 1., 1.) for x in range(color_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

# Generate statistics 
def image_stats(image, bboxes):
    p_bboxes = [box for box in bboxes if box[5] == 0]
    thresh = 50
    people = []
    stats = {
                "peopleNumber": {
                    "type": "Property",
                    "value": 0,
                    "observedAt": get_current_time()
                },
                "groupsOf2": {
                    "type": "Property",
                    "value": 0,
                    "observedAt": get_current_time()
                },
                "groupsOf3": {
                    "type": "Property",
                    "value": 0,
                    "observedAt": get_current_time()
                },
                "groupsOf4": {
                    "type": "Property",
                    "value": 0,
                    "observedAt": get_current_time()
                },
                "groupsOfMore": {
                    "type": "Property",
                    "value": 0,
                    "observedAt": get_current_time()
                }
            }
        
    # Get detections of people only
    for box in p_bboxes:
        w, h, _ = image.shape
        bw = box[2] - box [0]
        bh = box[3] - box [1]
        if bw*bh < w*h*0.1:
            xc = box[0] + bw/2
            yc = box[1] + bh/2
            people.append([xc,yc])
    
    # Get number of people
    stats["peopleNumber"]["value"] = len(people)
    
    
    # Group people based on image location
    if len(people) > 1:
        clustering = AgglomerativeClustering(n_clusters=None,compute_full_tree=True,distance_threshold = thresh).fit(people)
        group_labels = clustering.labels_

        groups = Counter(group_labels)
        for group in groups:
            if groups[group] == 2:
                stats["groupsOf2"]["value"] += 1
            elif groups[group] == 3:
                stats["groupsOf3"]["value"] += 1
            elif groups[group] == 4:
                stats["groupsOf4"]["value"] += 1
            elif groups[group] > 4:
                stats["groupsOfMore"]["value"] += 1

        # fill p bboxes
        for i in range(0,len(p_bboxes) - 1):
            p_bboxes[i][5] = group_labels[i]
        
    return json.dumps(stats), p_bboxes
    
# Update stellio entity
def update_entity(data):
    url = "contect broker"
    headers = {'Authorization': "",'Content-Type': 'application/json'}
    response = requests.patch(url, data = data, headers = headers)
    return response