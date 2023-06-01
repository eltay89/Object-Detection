# YOLO Object Detection with OpenCV
This code uses OpenCV and a pre-trained YOLOv3 model to detect objects in images. It allows a user to select an image through a file dialog, and then detects objects in that image using the YOLO model.

## How it works

### Download the YOLO model files
First, the code downloads the pre-trained YOLOv3 model files, including the weights and configuration file.

```python
# Download the pre-trained YOLOv3 model files 
urllib.request.urlretrieve("https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg", "yolov3.cfg")
urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
```

### Load the YOLO model 
Then, it loads the YOLO model using OpenCV.

```python
# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

### Get output layers 
It gets the names of the output layers of the model.

```python
# Determine the output layer names manually 
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

### Get user input image
It uses a file dialog to get an input image selected by the user.

```python
# Open a file dialog for the user to select an image
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
```

### Preprocess the image
It reads the image and resizes it to the required 416x416 input size for YOLO. It then normalizes the image values to be between 0 and 1.

```python
# Read the selected image
image = cv2.imread(file_path)

# Create a blob from the image (preprocessing step for YOLO)
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False) 
```

### Get detections 
It runs the preprocessed image through the YOLO model to get detections.

```python
# Set the input for the neural network
net.setInput(blob)

# Run forward pass through the network 
outs = net.forward(output_layers)
```

### Postprocess the detections
It filters out detections below a confidence threshold of 0.5. It then uses non-maximum suppression to filter out redundant detections.

```python
# Filter out weak detections by confidence 
if confidence > 0.5:

# Apply non-maximum suppression to remove redundant overlapping boxes  
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)   
```

### Draw detections
It draws the detections on the original input image, including the class label, confidence, and bounding box.

```python 
# Draw boxes and labels on the image for the detected objects
for i in indices:
    x, y, w, h = boxes[i]
    label = str(class_ids[i])
    confidence = confidences[i]

    # Draw the bounding box 
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add the label and confidence value as text 
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, f"{confidence:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### Show the output
It shows the final output image with the detections.

```python
# Show the analyzed image with boxes
cv2.imshow("Analyzed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
```
