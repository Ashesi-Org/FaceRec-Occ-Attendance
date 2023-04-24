from cams import Attendance as atd
import streamlit as st

# st.set_page_config(page_icon='🎒', page_title='Face Attendance', layout="wide")
st.set_page_config(page_icon='icon.ico', page_title='Face Attendance', layout="wide")


embeddings = "embeddings/face_embeddings.npz"
model = "models/students_model_v1.pkl"
database = "database/records.csv"

ashesi = atd(database=database,
             model=model,
             face_embeddings=embeddings)

ashesi.home()
