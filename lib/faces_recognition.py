import cv2
import numpy as np
from PIL import Image
import face_recognition


class FacesRecognition:
    def __init__(self):
        self.process = True
        self.known_encodings = []
        self.known_names = []

        self._default_encodings = [
            "images/jeova.jpg",
            "images/carina.jpg"
        ]

        self._default_names = [
            "Jeova Ramos",
            "Carina Mendes"
        ]

        for name, file in zip(self._default_names, self._default_encodings):
            self.add_face_encoders(file, name)

        return None

    def add_face_encoders(self, file, name):

        img = Image.open(file)
        face_encoding = face_recognition.face_encodings(
            np.asarray(img))[0]
        self.known_encodings.append(face_encoding)
        self.known_names.append(name)

        return None

    def _quarter_size(self, img):
        return cv2.resize(img, (0, 0), fx=.25, fy=.25)

    def _scale_back(location):
        (top, right, bottom, left) = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        return top, right, bottom, left

    def _bgr2rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _locations_encodings(self, img):
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, locations)

        return locations, encodings

    def _match_face(self, face_encoding):
        return face_recognition.compare_faces(
            self.known_encodings, face_encoding)

    def _closiest_face(self, face_encodings):
        distances = face_recognition.face_distance(
            self.known_encodings, face_encodings)

        return np.argmin(distances)

    def _process(self, img):
        locations, encodings = self._locations_encodings(img)

        names = []
        for face_encoding in encodings:
            matches = self._match_face(face_encoding)
            closiest = self._closiest_face(encodings)

            if matches[closiest]:
                name = self.known_names[closiest]
            else:
                name = "Unknown"

            names.append(name)

        return locations, encodings, names

    def recognize_faces(self, img, process=None):

        # Pre-process image
        img = self._quarter_size(img)
        img = self._bgr2rgb(img)

        process = self.process if process is None else process
        if process:
            locations, encodings, names = self._process(img)

        self.process = not process

        return locations, encodings, names

    def display_faces(self, img, locations, names):
        for location, name in zip(locations, names):
            (top, right, bottom, left) = self._scale_back(location)

            # Draw a box around the face
            cv2.rectangle(
                img, (left, top), (right, bottom),
                (0, 0, 255), 2
            )

            # Draw a label with a name below the face
            cv2.rectangle(
                img, (left, bottom - 35), (right, bottom),
                (0, 0, 255), cv2.FILLED
            )

            # Insert text into the image
            cv2.putText(
                img, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1
            )

        return img


if __name__ == "__main__":
    pass
