import cv2
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


class ImageDetection:
    def __init__(self):
        self.class_names, self.colors = self.load_labels()

        return None

    def three_lists_generator(self):
        boxes = []
        confidences = []
        class_ids = []

        return boxes, confidences, class_ids

    def load_labels(self, labels_path='models/cfg/coco.names'):
        # Load the COCO class names
        with open(labels_path, 'r') as f:
            class_names = f.read().strip().split('\n')

        # Get a different colors for each of the classes
        np.random.seed(42)
        colors = np.random.randint(
            0, 255,
            size=(len(class_names), 3),
            dtype='uint8')

        return class_names, colors

    def load_model(
        self,
        model_path='models/yolov4-tiny.weights',
        cfg_path='models/cfg/yolov4-tiny.cfg'
    ):
        net = cv2.dnn.readNet(
            model=model_path,
            config=cfg_path)

        return net

    def show_img(self, img):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def reshape_image(self, image, max_width=600):
        if image.shape[1] > max_width:
            image = cv2.resize(
                image,
                (
                    max_width,
                    int(image.shape[0] * max_width / image.shape[1]
                        )
                )
            )
        return image

    def load_image(self, img_path):
        image = cv2.imread(img_path)
        image = self.reshape_image(image)
        return image, image.shape[:2]

    def webcam_image(self, cap: cv2.VideoCapture):
        image = cap.read()[1]
        # image = self.reshape_image(image)

        return image, image.shape[:2]

    def blob_image(self, net, image):
        start = time.time()

        ln = net.getLayerNames()
        ln = [ln[ii[0] - 1] for ii in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            size=(416, 416),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        layer_outputs = net.forward(ln)
        end = time.time()

        fps = 1 / (end - start)

        return layer_outputs, fps

    def detections(
        self,
        detection, boxes, confidences,
        class_ids, H, W, threshold=0.5
    ):

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

        return boxes, confidences, class_ids

    def nms_objects(
        self,
        boxes, confidences,
        threshold=0.5, threshold_nms=0.3
    ):

        objects = cv2.dnn.NMSBoxes(
            boxes, confidences, threshold, threshold_nms)
        return objects

    def draw_bounding_boxes(
        self,
        objects, image, boxes,
        confidences, class_ids
    ):

        if len(objects) > 0:
            for ii in objects.flatten():
                (x, y) = (boxes[ii][0], boxes[ii][1])
                (w, h) = (boxes[ii][2], boxes[ii][3])
                color = [int(c) for c in self.colors[class_ids[ii]]]

                fill_area = np.full(
                    (image.shape), (0, 0, 0), dtype=np.uint8
                )

                text = f'{self.class_names[class_ids[ii]]}: '
                f'{confidences[ii]:.2f}'
                cv2.putText(
                    fill_area, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

                fx, fy, fw, fh = cv2.boundingRect(
                    fill_area[:, :, 2]
                )

                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, -1)
                cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, 3)

                cv2.putText(
                    image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1
                )

                # TODO: Save detections for specifc class
                # detection = image_cp[y:y+h, x:x+w]

                # TODO: Send a text message to the user if
                # the object is a person or pre defined class

            return image
        else:
            # print('No detections')
            return image

    def show_fps(self, image, fps):
        cv2.putText(
            img=image,
            text=f'{fps:.2f} FPS', org=(20, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2
        )

        return image

    def main(self):
        st.title("Webcam live objects detection")
        run = st.checkbox('Open Webcam')

        # Webcam
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        net = self.load_model()

        threshold = 0.5
        threshold_nms = 0.3

        while run:

            image, (H, W) = self.webcam_image(cap)
            layer_outputs, fps = self.blob_image(net, image)

            boxes, confidences, class_ids = self.three_lists_generator()
            for output in layer_outputs:
                for detection in output:
                    boxes, confidences, class_ids = self.detections(
                        detection,
                        boxes, confidences, class_ids,
                        H, W,
                        threshold=threshold
                    )

            # Non-maximum suppression
            objects = self.nms_objects(
                boxes, confidences,
                threshold=threshold, threshold_nms=threshold_nms
            )

            # Draw bounding boxes
            image = self.draw_bounding_boxes(
                objects, image,
                boxes, confidences, class_ids)

            # Show FPS
            image = self.show_fps(image, fps)
            # cv2.imshow('image', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
