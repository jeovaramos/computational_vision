import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        dectionCon: float = 0.5,
        trackCon: float = 0.5
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.dectionCon = dectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.dectionCon, self.trackCon
        )

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=False):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return lmList


def main():

    detector = HandDetector()
    while cap.isOpened():
        start = time.time()

        img = cap.read()[1]
        img = detector.findHands(img)
        detector.findPosition(img, 0, False)

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
