# Facial Detector

A Python-based program for detecting and matching objects in images using techniques such as:

- Gaussian Blurring for noise reduction
- Canny Edge Detection for boundary extraction
- Contour and Circle Detection using Hough Transform
- HOG (Histogram of Oriented Gradients) for shape and feature matching

## Features

1. **Multi-Scale Template Matching**: Matches a template object to regions in a larger search image.
2. **HOG Feature Extraction**: Identifies similarities based on shape and gradient orientation.
3. **Edge and Contour Analysis**: Displays intermediate processing steps for better understanding.
4. **Circular Shape Detection**: Highlights potential round objects using the Hough Circle Transform.

## Requirements

This program requires the following dependencies:
- Python 3.x
- OpenCV
- NumPy

You can install them using:
```
pip install opencv-python numpy
```

## How to Use

1. Clone this repository:
```
git clone https://github.com/ishaantek/facial-detector.git
cd facial-detector
```

2. Place the template image (`guy.png`) and the search image (`crowd.jpg`) in the images directory.

3. Run the program:
```
python main.py
```

4. The program will process the images and display:
   - The blurred image
   - The edge-detected image
   - Contours
   - Circles detected in the image
   - Final result with the best match highlighted (if found)

## Example Output

- **Template Preview**: Displays the object to match.
- **Blurred Search Image**: A pre-processed image to reduce noise.
- **Edge Map**: Detected edges in the image.
- **Contours and Circles**: Highlight shapes in the image.
- **Detection Result**: A rectangle around the best match with a confidence score.

## Files

- `object_match.py`: Main program file containing the implementation.
- `guy.png`: Template image to match (replace with your own).
- `crowd.jpg`: Search image for object detection.

## Repository Structure

```
FacialDetector/
├── main.py    # Main Python file
├── images/                # Folder with all images
│   ├── face.png           # Template image
│   └── group.jpg          # Search image
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! If you'd like to add features or improve the code, feel free to submit a pull request.
