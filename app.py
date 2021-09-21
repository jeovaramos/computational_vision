import requests
import streamlit as st
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


if __name__ == "__main__":
    header()
    hello_card()
    ImageDetection().main()
