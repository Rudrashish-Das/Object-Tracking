from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import numpy as np
import subprocess, time, os

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def download_video_segment(url, output_filename, duration=1):
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", url,
            "-t", str(duration),
            "-c", "copy",
            output_filename
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        raise (e)


def delete_old_files(file_list):
    current_time = time.time()

    while file_list:
        filename = file_list[0]  # Get the first file in the list
        file_path = filename

        # Get the file creation time
        creation_time = os.path.getctime(file_path)

        # Calculate the age of the file in seconds
        file_age = current_time - creation_time

        # Check if the file is older than 20 seconds and delete it
        if file_age > 20:
            print(f"Deleting {filename} created {file_age:.2f} seconds ago.")
            os.remove(file_path)
            file_list.pop(0)  # Remove the first file from the list
        else:
            break  # Stop deleting if the first file is not older than 20 seconds

def periodic_file_deletion(file_list, interval_seconds):
    while True:
        delete_old_files(file_list)
        time.sleep(interval_seconds)
