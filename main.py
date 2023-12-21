import ultralytics
import sys
#import yolox
import os
#from typing import List
#import numpy as np

# settings
MODEL = "yolov8n.pt"

HOME = os.getcwd()

SOURCE_VIDEO_PATH = f"{HOME}/test_video.mp4"
print("Target File:", SOURCE_VIDEO_PATH)

sys.path.append(f"{HOME}/ByteTrack")

import yolox
from typing import List
import numpy as np

import supervision
print("supervision.__version__:", supervision.__version__)

from lib.helperutil import *
from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

TARGET_VIDEO_PATH = f"{HOME}/output.mp4"

video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# settings
LINE_START = Point(10, video_info.height // 2)
LINE_END = Point(video_info.width - 10, video_info.height // 2)

from tqdm import tqdm


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCounter instance
line_counter = LineCounter(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        # model prediction on single frame and conversion to supervision Detections
        results = model(source=frame,classes=CLASS_ID)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {(confidence*100):0.2f}%"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # updating line counter
        line_counter.update(detections=detections)

        print("Roney: ", detections.__len__())
        print("Roney: ", line_counter.in_count, " ", line_counter.out_count)

        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)


