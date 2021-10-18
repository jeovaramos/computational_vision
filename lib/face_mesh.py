import cv2
import time
import mediapipe as mp


class FaceDetector:
    def __init__(
        self,
        mode: bool = False,
        max_faces: int = 2,
        refine: bool = True,
        detectionCon: float = 0.5
    ):
        self.mode = mode
        self.max_faces = max_faces
        self.refine = refine
        self.dectionCon = detectionCon

        self.mpDrawStyles = mp.solutions.drawing_styles
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=2
        )

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.max_faces,
            refine_landmarks=self.refine,
            min_detection_confidence=self.dectionCon
        )

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=faceLMS,
                        connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.mpDraw.DrawingSpec(
                            thickness=1, circle_radius=1, color=(0, 255, 0)
                        ),
                        connection_drawing_spec=self.mpDraw.DrawingSpec(
                            thickness=1, circle_radius=1
                        )
                    )

        return img


def main():
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start = time.time()

        img = cap.read()[1]
        img = detector.findFaceMesh(img)

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
