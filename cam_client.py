import os
import cv2
from darknet import darknet
import numpy as np
import math
import subprocess

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class Client:
    """
    The camera client component of the experiment.
    Responsible for:
    1. run object detection (YoloV4) and tracking (Deep SORT) and send results;
    2. read the images from dataset and convert to h264 video
    3. convert the video to request bitrate and send to the server

    """
    def __init__(self, id, batch_size = 15, dataset_dir='../others/dds/dataset/WildTrack/src/C'):
        # id: string
        self.id = id
        # video batch size integer
        self.batch_size = batch_size

        # displacement
        self.displacement_check = {}

        # Deep SORT encoding. setting is the same for now
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        self.temp_dir = 'temp-cropped'
        os.makedirs(self.temp_dir, exist_ok = True)



        self.dataset_dir = dataset_dir + id

        # read the total number of file from the server
        fnames = sorted(os.listdir(dataset_dir + id))
        self.total_frame = len(fnames)
        print("Total number of frames: ", str(self.total_frame))
        print("Simulating the camera with video frame size 15")
        # initiate the yolo v4 network
        network, class_names, class_colors = darknet.load_network(
            './darknet/cfg/yolov4.cfg',
            './darknet/cfg/coco.data',
            './darknet/yolov4.weights',
            batch_size=1
        )

        self.network = network
        self.class_names = class_names
        # initiate the deep sort network
        # multi-person tracking
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        print("Camera initiated")


    def first_phase(self, start_id):
        # read the images batch and run detections
        end_id = min(self.total_frame, int(start_id) + self.batch_size)
        print(end_id)
        total_obj = 0
        unique_obj_bbox = {}
        displacement_check = self.displacement_check

        for i in range(int(start_id), end_id):
            # print(self.dataset_dir + "/" + f"{str(i).zfill(10)}.png")
            image = cv2.imread(self.dataset_dir + "/" + f"{str(i).zfill(10)}.png")
            darknet_image = darknet.make_image(1920, 1080, 3)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())

            # detections list of tuple: (class_name, confidence_score, (bbox_info))
            detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=0.4)

            total_obj = total_obj + len(detections)
            bboxes = [obj[2] for obj in detections]
            confidence = [obj[1] for obj in detections]
            classes = [obj[0] for obj in detections]

            features = self.encoder(image_rgb, bboxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(bboxes, confidence, classes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            self.tracker.predict()
            self.tracker.update(detections)

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()

                if track.track_id not in unique_obj_bbox:
                    # unique_obj_bbox.append(track.track_id)
                    unique_obj_bbox[track.track_id] = {'feature': track.features[0].tolist()}
                    unique_obj_bbox[track.track_id]['length'] = 0

                # find the center point of the tracking object
                center_point = track.to_tlwh()
                c_x = center_point[0] + (center_point[2]) / 2

                c_y = center_point[1] + (center_point[3]) / 2

                if track.track_id not in displacement_check:
                    displacement_check[track.track_id] = (c_x, c_y)
                else:
                    disp = math.sqrt((c_x - displacement_check[track.track_id][0]) ** 2 + (c_y - displacement_check[track.track_id][1]) ** 2)
                    # print(unique_obj_bbox[track.track_id])
                    # print('disp for cam: ', str(track.track_id), " ", str(disp))
                    unique_obj_bbox[track.track_id]['length'] = unique_obj_bbox[track.track_id]['length'] + disp

                    # update the center point for next iteration
                    displacement_check[track.track_id] = (c_x, c_y)
                # print(displacement_check)
                # print(unique_obj_bbox[track.track_id]['length'])

        self.displacement_check = displacement_check
        # print(displacement_check)
        return {'total_obj': total_obj, 'unique_obj_bbox': unique_obj_bbox}


    def second_phase(self, bitrate, start_id):
        encoded_vid_path = os.path.join(self.temp_dir, "temp.mp4")
        if not bitrate:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{self.dataset_dir}/%010d.png",
                                              "-vcodec", "libx264", "-g", "15",
                                              "-keyint_min", "15",
                                              "-pix_fmt", "yuv420p",
                                              "-frames:v",
                                              str(self.batch_size),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
        else:
            rate=str(bitrate)+"k"
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{self.dataset_dir}/%010d.png",
                                              "-vcodec", "libx264",
                                              "-g", "15",
                                              "-keyint_min", "15",
                                              "-maxrate", f"{rate}",
                                              "-b", f"{rate}",
                                              "-bufsize", f"{rate}",
                                              "-pix_fmt", "yuv420p",
                                              "-frames:v",
                                              str(self.batch_size),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)

        return "OK"
