import cv2
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie as stl
from lib.faces_recognition import FacesRecognition


st.set_page_config(
    page_title="Image detection - Jeová Ramos",
    page_icon=":eye:",
    layout="centered")


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


def form2add(fr: FacesRecognition, file, name, submitted):
    if submitted:
        fr.add_face_encoders(file, name, folder=False)
        st.write('Known faces', fr.known_names)
        st.write('Number of faces:', len(fr.known_encodings))
        submitted = False

    return fr


fr = FacesRecognition()


if __name__ == "__main__":
    header()
    # hello_card()

    # form = st.form(key='my-form')
    # name = form.text_input('Enter your name')
    # submit = form.form_submit_button('Submit')

    # st.write('Press submit to have your name printed below')

    # if submit:
    #     st.write(f'hello {name}')

    with st.sidebar:
        form = st.form(key='Register face')
        name = form.text_input("Enter your name")

        file = form.file_uploader(
            "Upload image", type=['png', 'jpeg', 'jpg'])

        submitted = form.form_submit_button('Submit')
    fr = form2add(fr, file, name, submitted)

    cap = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])
    run = st.checkbox('Open Webcam')

    key = True
    while run:

        start = time.time()
        img = cap.read()[1]

        locations, _, names = fr.recognize_faces(img, process=True)

        img = fr.display_faces(img, locations, names)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = print_fps(img, start)
        FRAME_WINDOW.image(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
