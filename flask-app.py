import ultralytics
import sys
import yolox
import os
from typing import List
import numpy as np
import supervision
from lib.helperutil import *
from ultralytics import YOLO
from tqdm.auto import tqdm
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import cv2
import threading
import time
# import random

# settings
MODEL = "yolov8n.pt"

HOME = os.getcwd()

model = YOLO(MODEL)
model.fuse()

MOD = 10e9
DOWNLOAD_DURATION = 10 # seconds
TIMEOUT = 30 # seconds

SOURCE_VIDEO_PATH = f"{HOME}/test_video.mp4"
TARGET_VIDEO_PATH = f"{HOME}/output.mp4"

#STREAM_URL = 'https://devimages.apple.com.edgekey.net/streaming/examples/bipbop_16x9/bipbop_16x9_variant.m3u8'
STREAM_URL = "https://live.igdrones.com:8080/hls/test.m3u8"
#STREAM_URL = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
#STREAM_URL = "https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8"

STREAM = True

HOST="0.0.0.0"
PORT=8080

sys.path.append(f"{HOME}/ByteTrack")

print("supervision.__version__:", supervision.__version__)

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0]

# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0, text_padding=0)
line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# random.seed(int(time.time()))

in_room = 0;
in_count = 0;
out_count = 0;
obj_in_frame = 0;


# Set the interval for running the file deletion function (5 seconds in this case)
deletion_interval = 18

# Start a separate thread for periodic file deletion
# deletion_thread = threading.Thread(target=periodic_file_deletion, args=(video_serial, deletion_interval), daemon=True)
# deletion_thread.start()

background_thread_running = False
latest_frame = None
# lock = threading.Lock()

connected = 0
last_called = 0 # time.time()
#time.sleep(TIMEOUT + 5)
video_error = None

def object_tracking(source, output="output.mp4", stream=False, random_out=False):
    global video_error, in_room, in_count, out_count, obj_in_frame, background_thread_running, iteration_time, latest_frame, connected, last_called
    running = True
    stream_url = source
    try:
        while running:

            if (connected <= 0) and (time.time() - last_called >= TIMEOUT):
                time.sleep(5)
                print("IDLE...")
                video_error = None
                continue

            if source is None:
                raise Exception("No source file given!")

            try:
                if stream:
                    download_video_segment(stream_url,"cache.mp4",duration=DOWNLOAD_DURATION)
                    source = 'cache.mp4'
                    video_error = False
            except Exception as e:
                print('Error downloading video segment ', connected)
                video_error = True
                time.sleep(5)
                continue

            running = stream

            print("Target File:", source)

            video_info = VideoInfo.from_video_path(source)

            # settings
            LINE_START = Point(10, video_info.height // 2)
            LINE_END = Point(video_info.width - 10, video_info.height // 2)

            # create frame generator
            generator = get_video_frames_generator(source)
            # create LineCounter instance
            line_counter = LineCounter(start=LINE_START, end=LINE_END)

            try:
                # loop over video frames
                for frame in tqdm(generator, total=video_info.total_frames, disable=True):
                    # if connected <= 0:
                    #     break

                    start_time = time.time()
                    # model prediction on single frame and conversion to supervision Detections
                    results = model(source=frame,classes=CLASS_ID, verbose=False)
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
                        #f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {(confidence*100):0.1f}%"
                        f"{(confidence*100):0.1f}%"
                        for _, confidence, class_id, tracker_id
                        in detections
                    ]
                    # updating line counter
                    line_counter.update(detections=detections)

                    obj_in_frame = detections.__len__()
                    in_room += abs(in_count  - line_counter.in_count % MOD)
                    in_room = in_room - abs(out_count  - line_counter.out_count % MOD) if in_room - abs(out_count - line_counter.out_count % MOD) > 0 else 0

                    in_count = line_counter.in_count % MOD
                    out_count = line_counter.out_count % MOD

                    # annotate and display frame
                    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                    line_annotator.annotate(frame=frame, line_counter=line_counter)
                    # with lock:
                    latest_frame = frame

                    end_time = time.time()  # Record the end time of the iteration
                    iteration_time = end_time - start_time

            except Exception as e:
                print(f"An error occurred: {e}")
            # time.sleep(5)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        background_thread_running = False

# Start the run() method in a separate thread when the Flask app starts
background_thread = threading.Thread(target=object_tracking,args=[STREAM_URL, TARGET_VIDEO_PATH, True])
background_thread.start()
background_thread_running = True

app = Flask(__name__)
CORS(app, support_credentials=True)

def generate():
    global connected
    connected += 1
    try:
        while True:
            if video_error == True:
                break
            # with lock:
            if latest_frame is None:
                continue

            ret, jpeg = cv2.imencode('.jpg', latest_frame)

            # Yield the frame in bytes
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
    except Exception as e:
        print(e)
    finally:
        connected -= 1

# Route for streaming annotated frames
@app.route('/annotated')
def annotated_frames():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/read', methods=['GET'])
@cross_origin(supports_credentials=True)
def read_data():
    global last_called
    last_called = time.time()

    try:
        content = {
                    'in_room': in_room,
                    'in_count': in_count,
                    'out_count': out_count,
                    'obj_in_frame': obj_in_frame
                }
        if video_error == None:
            return jsonify({'success': False, 'error': "Initializing video stream! Please Retry!"})
        if video_error == True:
            return jsonify({'success': False, 'error': "Error with source stream!"})
        return jsonify({'success': True, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Start the Flask app
    app.run(host=HOST, port=PORT)
