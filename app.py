import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
#title
st.title('Transportni klassifikatsiya qiluvchi model')

#rasmni joylash
file=st.file_uploader('Rasm yuklash',type=['png','jpeg','jpg','SVG','jfif'])

if  file:
    st.image(file)
    #PIL convert
    img=PILImage.create(file)
    #model
    modelpath=r'\Users\ASUS\Desktop\github_upload\transportclassification\transport_model.pkl'
    model = load_learner(modelpath)


    #prediction
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimolik: {probs[pred_id]*100:.1f}%")

    #plotting
    fig=px.bar(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)
