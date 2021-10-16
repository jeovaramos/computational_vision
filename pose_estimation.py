import cv2
import time
import mediapipe as mp


class PoseDetector:
    def __init__(
        self,
        mode: bool = False,
        upBody: bool = False,
        smooth: bool = True,
        detectionCon: float = 0.5,
        trackCon: float = 0.5
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.dectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.upBody,
            smooth_segmentation=self.smooth,
            min_detection_confidence=self.dectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS
                )

        return img

    def findPosition(self, img, draw=False):

        lmList = []
        if self.results.pose_landmarks:
            myHand = self.results.pose_landmarks
            for id, lm in enumerate(myHand.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return lmList


def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start = time.time()

        img = cap.read()[1]
        img = detector.findPose(img)
        detector.findPosition(img, False)

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
