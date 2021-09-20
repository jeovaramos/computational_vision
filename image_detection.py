import cv2
import time
import numpy as np

# Load the COCO class names
with open('models/cfg/coco.names', 'r') as f:
    class_names = f.read().strip().split('\n')

# Get a different colors for each of the classes
np.random.seed(42)
colors = np.random.randint(
    0, 255,
    size=(len(class_names), 3))

# Load the DNN model
model = cv2.dnn.readNet(
    model='models/yolov4.weights',
    config='models/cfg/yolov4.cfg')

# Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Webcam
cap = cv2.VideoCapture(0)

threshold = 0.5
threshold_nms = 0.3

while cap.isOpened():
    boxes = []
    confidences = []
    class_ids = []

    # Read in the image
    success, image = cap.read()
    (H, W) = image.shape[:2]

    ln = model.getLayerNames()
    ln = [ln[ii[0] - 1] for ii in model.getUnconnectedOutLayers()]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        size=(416, 416),
        # mean=(104, 117, 123),
        swapRB=True,
        crop=False
    )

    # start time to calculate FPS
    start = time.time()

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    layer_outputs = model.forward(ln)

    # End time
    end = time.time()

    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    # Run over each of the detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    objects = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        threshold,
        threshold_nms
    )

    # Draw bounding boxes
    if len(objects) > 0:
        for ii in objects.flatten():
            (x, y) = (boxes[ii][0], boxes[ii][1])
            (w, h) = (boxes[ii][2], boxes[ii][3])
            color = [int(c) for c in colors[class_ids[ii]]]

            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text = f'{class_names[class_ids[ii]]}: {confidences[ii]:.2f}'

            cv2.putText(
                image,
                text,
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # Show FPS
    cv2.putText(
        image,
        f"{fps:.2f} FPS", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow('image', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
