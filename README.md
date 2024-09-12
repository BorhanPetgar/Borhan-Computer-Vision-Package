# Computer Vision Utility Package

Welcome to my personal Computer Vision utility package! This repository contains a collection of common functions to facilitate various tasks in computer vision, such as extracting frames from videos, creating videos from frames, and more.

## Features

- **Extract Frames from Video**: Efficiently extract specified frames from a video and save them as images.
- **Create Video from Frames**: Combine a series of images into a video file.
- **Get Video Information**: Retrieve essential properties of a video, including FPS, dimensions, and total frame count.
- **Frame Sorting**: Sort frames in a human-friendly manner.

## Installation

To use this package, ensure you have Python and the required libraries installed. You can install the dependencies using pip:

```bash
pip install opencv-python tqdm
```

## Usage

### Extract Frames from Video

To extract frames from a video, use the `extract_frames` function:

```python
extract_frames(input_video_path, output_video_dir, start_frame=None, end_frame=None, show_log=True)
```

- **input_video_path**: Path to the input video file.
- **output_video_dir**: Directory where extracted frames will be saved.
- **start_frame**: Optional; start frame index (default is 0).
- **end_frame**: Optional; end frame index (default is total frames).
- **show_log**: Optional; whether to display logs during extraction (default is True).

### Create Video from Frames

To create a video from a directory of frames, use the `create_video` function:

```python
create_video(input_frames_dir, output_video_path, fps=30)
```

- **input_frames_dir**: Directory containing the frames.
- **output_video_path**: Path where the output video will be saved.
- **fps**: Optional; frames per second for the output video (default is 30).

### Get Video Information

To retrieve video properties, use the `get_video_info` function:

```python
video_info = get_video_info(video_path)
```

- **video_path**: Path to the video file.
- Returns a dictionary with `fps`, `width`, `height`, and `total_frames`.

### Frame Sorting

The package includes a helper function `sorted_nicely` to sort frame filenames in a natural order.

## Future Ideas

- Implement conversion between bounding box formats (xyxy ↔ xywh).
- Add YOLO inference functionalities and tune parameters for better performance.

## Contributing

Feel free to fork the repository and submit pull requests if you have improvements or additional features!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the OpenCV and Ultralytics communities for their amazing libraries that make computer vision tasks easier!
