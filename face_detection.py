import cv2
import time
import mediapipe as mp


class FaceDetector:
    def __init__(
        self,
        detectionCon: float = 0.5,
    ):
        self.dectionCon = detectionCon

        self.mpFaceDetection = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.face_detection
        self.FaceDetection = self.mpPose.FaceDetection(
            model_selection=0, min_detection_confidence=detectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.FaceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = img.shape[:2]
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)

                cv2.putText(
                    img=img,
                    text=f"{int(detection.score[0] * 100)}%",
                    org=(bbox[0], bbox[1] - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 255),
                    thickness=2
                )

        return img

    def fancyDraw(self, img, bbox, ll=30, tt=5, rt=1):

        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top left
        cv2.line(img, (x, y), (x + ll, y), (255, 0, 255), tt)
        cv2.line(img, (x, y), (x, y + ll), (255, 0, 255), tt)

        # Top right
        cv2.line(img, (x1, y), (x1 - ll, y), (255, 0, 255), tt)
        cv2.line(img, (x1, y), (x1, y + ll), (255, 0, 255), tt)

        # Bottom left
        cv2.line(img, (x, y1), (x + ll, y1), (255, 0, 255), tt)
        cv2.line(img, (x, y1), (x, y1 - ll), (255, 0, 255), tt)

        # Bottom right
        cv2.line(img, (x1, y1), (x1 - ll, y1), (255, 0, 255), tt)
        cv2.line(img, (x1, y1), (x1, y1 - ll), (255, 0, 255), tt)

        return img


def main():
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start = time.time()

        img = cap.read()[1]
        img = detector.findFaces(img)
        # detector.findPosition(img, False)

        fps = 1 / (time.time() - start)

        cv2.putText(
            img=img,
            text=f"FPS: {fps:.2f}",
            org=(10, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2
        )

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
