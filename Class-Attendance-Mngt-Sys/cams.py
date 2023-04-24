import streamlit as st
from PIL.Image import Image
from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from streamlit_option_menu import option_menu
from datetime import date
import time


class Attendance:
    def __init__(self, database, face_embeddings, model):
        self.database = Attendance.__get_database__(database)
        self.embeddings = Attendance.__get_features__(face_embeddings)
        self.model = Attendance.__get_model__(model)

        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.encoder = LabelEncoder()

        self.encoder.fit(self.embeddings)
        self.encodings = self.encoder.transform(self.embeddings)

    @staticmethod
    #@st.cache_data
    def __get_model__(model):
        m = pickle.load(open(model, 'rb'))
        return m

    @staticmethod
    #@st.cache_data
    def __get_features__(emb):
        embed = np.load(emb)['arr_1']
        return embed

    @staticmethod
    def __get_database__(data):
        db = pd.read_csv(data)
        return db

    def home(self):
        # st.set_page_config(page_icon='🎒', page_title='CAMS', layout="wide")
        st.markdown("<h1 style='text-align: center'>CLASS ATTENDANCE MANAGEMENT SYSTEM</h1>", unsafe_allow_html=True)
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        with st.sidebar:
            st.header("Menu")
            option = option_menu('', ['📸 Take Attendance', '📊 Attendance Records'],
                                 icons=["nothing", "nothing"],
                                 menu_icon='nothing',
                                 orientation='vertical')
        if option == "📸 Take Attendance":
            face = Attendance.__get_face()
            detected_face = self.__detect_face__(face)
            self.__recognize_face__(detected_face)
        elif option == "📊 Attendance Records":
            st.header("📊 Attendance Records")
            st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            self.__display_database()
            reset = st.button("Reset Database")
            if reset:
                self.__reset_database__()
                # self.__display_database()

    @staticmethod
    def __get_face():
        st.header('Take Attendance')
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        cam, feedback = st.tabs(['✅ Take Attendance', '🤔Report an Issue'])
        with cam:
            with st.expander("Open Camera and take picture"):
                st.markdown("---")
                picture = st.camera_input("Enjoy class today and be great 😉✌️")

            if picture:
                img: Image = Image.open(picture)
                img_array = np.array(img)
                return img_array

    def __detect_face__(self, face):
        detected_face = None

        try:
            cv_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            x, y, w, h = self.detector.detect_faces(cv_face)[0]['box']
            detected_face = cv_face[y:y + h, x:x + w]
            detected_face = cv2.resize(detected_face, (160, 160))
            detected_face = np.expand_dims(detected_face, axis=0)
            # st.image(detected_face)
        except:
            pass

        return detected_face

    def __recognize_face__(self, detected_face):
        if detected_face is not None:
            face_embedding = self.facenet.embeddings(detected_face)
            # probability = self.model.predict_proba(face_embedding)
            prediction = self.model.predict(face_embedding)
            student_name = self.encoder.inverse_transform(prediction)[0]
            st.success(f"✅ Attendance for {student_name} has been marked.")
            self.__update_database__(student_name)
        else:
            st.warning("No face detected yet")

    def __update_database__(self, name):
        # df.loc[df['name'] == student_name, 'status'] = 'Yes'
        self.database.loc[self.database['Student Name'] == name, 'Status'] = "Present 📗"
        self.database.loc[self.database['Student Name'] == name, 'Date'], t = date.today(), time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        self.database.loc[self.database['Student Name'] == name, 'Time'] = current_time
        self.database.to_csv('database/records.csv', index=False, header=True)

    def __reset_database__(self):
        self.database['Status'] = "Absent 📕"
        self.database['Date'] = ''
        self.database['Time'] = ''
        self.database.to_csv('database/records.csv', index=False, header=True)

    def __display_database(self):
        st.dataframe(self.database, use_container_width=True)
