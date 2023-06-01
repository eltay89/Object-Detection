import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import urllib.request

# Download the pre-trained YOLOv3 model files
#urllib.request.urlretrieve("https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg", "yolov3.cfg")
#urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")

# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Determine the output layer names manually
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create a function to handle the button click event
def analyze_image():
    # Open a file dialog for the user to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    # Read the selected image
    image = cv2.imread(file_path)

    # Get the image dimensions
    height, width, channels = image.shape

    # Create a blob from the image (preprocessing step for YOLO)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

    # Set the input for the neural network
    net.setInput(blob)

    # Run forward pass through the network
    outs = net.forward(output_layers)

    # Initialize lists for detected objects' class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each output layer

    # outs likely only has a single result
    out = outs[0]
    for detection in out:
        for detection in out:
            # Get the class probabilities and find the most confident class
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions by confidence threshold
            if confidence > 0.5:
                # Scale the bounding box coordinates to the original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add the class ID, confidence, and bounding box coordinates to the lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)



    # Draw boxes and labels on the image for the detected objects
    for i in indices:
        # Get the coordinates of the bounding box
        x, y, w, h = boxes[i]
        label = str(class_ids[i])
        confidence = confidences[i]

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add the label and confidence value as text
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the analyzed image with boxes
    cv2.imshow("Analyzed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create a Tkinter window
window = tk.Tk()

# Create a button widget
button = tk.Button(window, text="Select Image", command=analyze_image)

# Add the button to the window
button.pack()

# Create a close button
button2 = tk.Button(window, text="Close", command=window.quit)
button2.pack()

# Start the Tkinter event loop
window.mainloop()
