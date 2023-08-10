# import modules
import cv2  # pip install opencv-python
import numpy as np # pip install numpy

# YOLOv3 model is loaded
net = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg") # can choose any yolov3 model (path) of your choice
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# webcam is loaded
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # a blob is created from the webcam frame and then passes it through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)

    # the lists are initialized to store detection information
    class_ids = []
    confidences = []
    boxes = []

    # each layer is then processed
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # bounding boxes are drawn and labels with colors
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection", frame) # output webcam video is displayed
    if cv2.waitKey(1) & 0xFF == ord('s'): # key S is used to break the loop
        break

cap.release()  # the webcam is released
cv2.destroyAllWindows()  # all opencv windows are destroyed
