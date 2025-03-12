import os
import cv2
import re
from tqdm import tqdm
import numpy as np

def create_patches(image, patch_size=100, overlap=0):
    """
    Split image into patches of specified size with optional overlap.
    
    Args:
        image: Input image (numpy array)
        patch_size: Size of each patch (square)
        overlap: Overlap between adjacent patches in pixels
        
    Returns:
        Tuple of (patches, patch_positions) where patches is a list of image patches
        and patch_positions is a list of (x, y) coordinates for top-left corner of each patch
    """
    height, width = image.shape[:2]
    patches = []
    patch_positions = []
    
    stride = patch_size - overlap
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size].copy()
            patches.append(patch)
            patch_positions.append((x, y))
    
    return patches, patch_positions


def draw_boxes(image, boxes, classes=None, confidences=None, colors=None, class_names=None):
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (numpy array)
        boxes: List of boxes in format [x1, y1, x2, y2]
        classes: List of class IDs for each box (optional)
        confidences: List of confidence scores for each box (optional)
        colors: Dictionary mapping class IDs to colors, or a single color for all boxes
        class_names: Dictionary mapping class IDs to class names
        
    Returns:
        Image with drawn bounding boxes
    """
    image_with_boxes = image.copy()
    
    if colors is None:
        # Generate random colors if not provided
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Determine color
        if isinstance(colors, dict) and classes is not None:
            color = colors.get(classes[i], (0, 255, 0))
        elif isinstance(colors, (list, np.ndarray)) and classes is not None:
            color = tuple(map(int, colors[classes[i] % len(colors)]))
        else:
            color = (0, 255, 0) if not isinstance(colors, tuple) else colors
        
        # Draw box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if class information is available
        if classes is not None:
            cls_id = classes[i]
            
            label_parts = []
            
            # Add class name if available
            if class_names and cls_id in class_names:
                label_parts.append(f"{class_names[cls_id]}")
            else:
                label_parts.append(f"Class {cls_id}")
                
            # Add confidence if available
            if confidences is not None:
                label_parts.append(f"{confidences[i]:.2f}")
                
            label = ": ".join(label_parts)
            
            # Position the label
            y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(image_with_boxes, label, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_with_boxes


def run_yolo_inference(model, image, conf_threshold=0.25, patch_size=None, overlap=0):
    """
    Run YOLO inference on an image, with optional patching for large images.
    
    Args:
        model: YOLO model instance from ultralytics
        image: Input image (numpy array)
        conf_threshold: Confidence threshold for detections
        patch_size: If not None, split image into patches of this size
        overlap: Overlap between patches in pixels (only if patch_size is specified)
        
    Returns:
        Tuple of (boxes, classes, confidences) where:
            boxes: numpy array of bounding boxes in [x1, y1, x2, y2] format
            classes: numpy array of class IDs
            confidences: numpy array of confidence scores
    """
    if patch_size is None:
        # Run on the whole image
        results = model(image)
        
        boxes = []
        classes = []
        confidences = []
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    if box.conf >= conf_threshold:
                        boxes.append(box.xyxy.cpu().numpy()[0])
                        classes.append(int(box.cls.cpu().numpy()[0]))
                        confidences.append(float(box.conf.cpu().numpy()[0]))
        
        return (np.array(boxes) if boxes else np.empty((0, 4))), \
               (np.array(classes) if classes else np.array([])), \
               (np.array(confidences) if confidences else np.array([]))
    else:
        # Run on patches
        patches, patch_positions = create_patches(image, patch_size, overlap)
        
        all_boxes = []
        all_classes = []
        all_confidences = []
        
        for i, patch in enumerate(patches):
            x_offset, y_offset = patch_positions[i]
            
            # Run inference on the patch
            results = model(patch)
            
            for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.conf >= conf_threshold:
                            # Get box coordinates and adjust to original image
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            
                            x1 += x_offset
                            y1 += y_offset
                            x2 += x_offset
                            y2 += y_offset
                            
                            all_boxes.append([x1, y1, x2, y2])
                            all_classes.append(int(box.cls.cpu().numpy()[0]))
                            all_confidences.append(float(box.conf.cpu().numpy()[0]))
        
        return (np.array(all_boxes) if all_boxes else np.empty((0, 4))), \
               (np.array(all_classes) if all_classes else np.array([])), \
               (np.array(all_confidences) if all_confidences else np.array([]))
               

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply non-maximum suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for considering boxes as duplicates
        
    Returns:
        List of indices of boxes to keep
    """
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays if they aren't already
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if isinstance(scores, list):
        scores = np.array(scores)
    
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area of each box
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Pick the box with highest confidence
        i = order[0]
        keep.append(i)
        
        # Find IoU with rest of the boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Compute width and height of intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute IoU
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


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

