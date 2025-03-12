import os
import cv2
import re
from tqdm import tqdm
import numpy as np


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": width, "height": height, "total_frames": total_frames}


def extract_frames(input_video_path, output_video_dir, start_frame=None, end_frame=None, show_log=True):
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Get video properties
    fps, width, height, total_frames = get_video_info(input_video_path).values()
    print(f'Video FPS: {fps}, Width: {width}, Height: {height}, Total frames: {total_frames}')
    
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames
        
    assert start_frame < end_frame, "Start frame must be less than end frame"
    assert end_frame <= total_frames, "End frame must be less than or equal to total frames"
    
    cap = cv2.VideoCapture(input_video_path)
    
    frame_count = 0
    with tqdm(total=(end_frame - start_frame), desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count >= start_frame and frame_count < end_frame:
                output_video_path = os.path.join(output_video_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(output_video_path, frame)
                if show_log:
                    print(f"Frame {frame_count} saved to {output_video_path}")
                    frame_count += 1
            pbar.update(1)
    cap.release()
    print(f"Frames saved to {output_video_dir}")


def create_video(input_frames_dir, output_video_path, fps=30):
    frames = []
    for frame_name in sorted_nicely(os.listdir(input_frames_dir)):
        frame_path = os.path.join(input_frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        frames.append(frame)
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in tqdm(frames, desc='Creating Video'):
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")
        

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Bounding box coordinate conversion functions
def xyxy_to_xywh(box):
    """
    Convert bounding box from [x1, y1, x2, y2] format to [x, y, width, height] format.
    
    Args:
        box: List or numpy array [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right
        
    Returns:
        List [x, y, width, height] where (x, y) is center point
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return [x_center, y_center, width, height]


def xywh_to_xyxy(box):
    """
    Convert bounding box from [x, y, width, height] format to [x1, y1, x2, y2] format.
    
    Args:
        box: List or numpy array [x, y, width, height] where (x, y) is center point
        
    Returns:
        List [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right
    """
    x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates of intersection
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2_min - x1_max)
    height = max(0, y2_min - y1_max)
    intersection_area = width * height
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    
    return intersection_area / union_area

