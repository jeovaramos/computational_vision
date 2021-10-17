import cv2
import time
import requests
import numpy as np
import streamlit as st
import face_recognition
from streamlit_lottie import st_lottie as stl
from lib.image_detection import ImageDetection

st.set_page_config(
    page_title="Image detection - Jeová Ramos",
    page_icon=":eye:",
    layout="centered", )


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None

    return r.json()


def header():
    lottie = load_lottieurl(
        'https://assets1.lottiefiles.com/private_files/lf30_hybmflns.json')
    stl(lottie, speed=1, height=200, key="initial")

    return None


def hello_card():
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (.1, 2, .2, 1.5, .1))

    row0_1.title('Analytics dashboard')
    with row0_2:
        st.write('')

    row0_2.subheader(
        'A Web App by [Jeová Ramos](https://github.com/jeovaramos)')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

    with row1_1:
        st.write(
            "Hey there! Welcome to my first Computational vision project. "
            "I am looking for build my data science portfolio, "
            "so I made this with studie intentions. "
            "I'm using **Open CV and YOLO v4** tiny model to identify objects."
            " The tiny version is way more light, faster and enought for this "
            "project propurse.")
        st.write(
            "If you want to keep in touch or just see other things "
            "I'm working in, please consider click in the link above "
            "with my name on it.")
        st.write(
            "**I hope you enjoy it.** Best regards,\n Jeová Ramos.")

    return None


jeova_image = face_recognition.load_image_file("images/jeova.jpg")
jeova_face_encoding = face_recognition.face_encodings(jeova_image)[0]

carina_image = face_recognition.load_image_file("images/carina.jpeg")
carina_face_encoding = face_recognition.face_encodings(carina_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jeova_face_encoding,
    carina_face_encoding

]
known_face_names = [
    "Jeova Ramos",
    "Carina Mendes"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process = True


def recognize_faces(img, process, face_locations, face_encodings, face_names):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process = not process

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6),
                    font, 0.7, (255, 255, 255), 1)

    return img, process, face_locations, face_encodings, face_names


def print_fps(img, start):
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
    return img


if __name__ == "__main__":
    header()
    hello_card()

    cap = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])
    run = st.checkbox('Open Webcam')

    while run:
        start = time.time()
        img = cap.read()[1]

        img, process, face_locations, face_encodings, face_names = \
            recognize_faces(
                img, process, face_locations, face_encodings, face_names
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = print_fps(img, start)
        FRAME_WINDOW.image(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
