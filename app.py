import streamlit as st
import streamlit.components.v1 as components
from multiapp import MultiApp
from apps import diagnose,diagnose_profile,home

app = MultiApp()

st.markdown(""" <style> .font {
font-size:30px ; font-family: 'Arial'; color:indigo; ; text-align:center} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font"><b>Diagnosis Prediction based on Clinical Text</b></p>', unsafe_allow_html=True)

# Add all your application here
app.add_app('Chief complaint', diagnose.app)
app.add_app('Chief complaint with Patient Profile', diagnose_profile.app)


# The main app
app.run()
