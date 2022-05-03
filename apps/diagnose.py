import streamlit as st
from multiapp import MultiApp

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd
import pythainlp
import pythainlp.util
from pythainlp.util import collate
import datetime
from pythainlp.util import thai_strftime
from pythainlp.util import thai_time
from pythainlp.util import num_to_thaiword, thaiword_to_num, bahttext
from pythainlp import sent_tokenize, word_tokenize
from pythainlp.tokenize.multi_cut import find_all_segment, mmcut, segment
from pythainlp import subword_tokenize
from pythainlp.tokenize import syllable_tokenize
from pythainlp.tokenize import Tokenizer
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
from pythainlp.spell import correct ,spell
from pythainlp.tag import pos_tag

from bokeh.models.widgets import Button


from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



Group_model = load_model('model/LSTM-5-class_model_smote.h5')
GI_model = load_model('model/GRU-GI_model.h5')
Cardio_model = load_model('model/LSTM-Cardio_model.h5')
Chest_model = load_model('model/LSTM-Chest_model.h5')
Neuro_model = load_model('model/LSTM-Neuro_model.h5')
Endocrine_model = load_model('model/LSTM-Endocrine_model.h5')
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer_group = pickle.load(handle)
with open('model/Chest_tokenizer.pickle', 'rb') as handle:
    tokenizer_chest = pickle.load(handle)
with open('model/GI_tokenizer.pickle', 'rb') as handle:
    tokenizer_gi = pickle.load(handle)
with open('model/Cardio_tokenizer.pickle', 'rb') as handle:
    tokenizer_cardio = pickle.load(handle)
with open('model/Neuro_tokenizer.pickle', 'rb') as handle:
    tokenizer_neuro = pickle.load(handle)
with open('model/Endocrine_tokenizer.pickle', 'rb') as handle:
    tokenizer_endocrine = pickle.load(handle)
symptoms=['ไข้','ไอ','เจ็บคอ','น้ำมูก','น้ำมูกใส','น้ำมูกเขียว','น้ำมูกข้น','เสมหะ','ไอแห้ง','กลืนลำบาก','คอแห้ง','เพลีย','น้ำมูก','ปวดเมื่อย','ปวดศีรษะ','ซึม','เวียนศรีษะ',
          'เสียงแหบ','หอบ','เจ็บหน้าอก','แน่นหน้าอก','เหนื่อย','หนาวสั่น','คัดจมูก','เหนื่อยหอบ','แสบร้อน','จุก','จุกคอ','จุกเสียด','เรอ','หอบหืด',
          'ลิ้นปี่','ท้องอืด','แน่นท้อง','แน่นอก','แน่นคอ','ปวดท้องน้อย','ถ่ายปกติ','ตัวร้อน','แน่นหน้าอกซ้าย','แน่นหน้าอกข้างซ้าย','แน่นหน้าอกขวา','แน่นหน้าอกข้างขวา',
          'ท้องผูก', 'ปวดท้องน้อย','ปวดกลางท้อง','ปวดท้องกลางท้อง','ปวดท้องตรงกลาง','ปวดท้องด้านซ้าย','จุกท้อง',
          'ปวดท้องข้างซ้าย', 'ท้องเสีย','เบื่ออาหาร', 'อาเจียน','ปวดท้อง','ปวดท้องรอบสะดือ','ปวดเมื่อย',
          'ปวดท้องด้านขวาล่าง','ปวดท้องข้างขวา','ปวดท้องด้านขวา','เสมหะปนเลือด',
          'อ่อนเพลีย','คลื่นไส้','ถ่ายเป็นน้ำ','ถ่ายเหลว','ถ่ายบ่อย','สะดือ','ปวดหัว','เจ็บท้อง',
           'อุจจาระบ่อย','ไข้ต่ำ','ไข้สูง','แสบคอ','ถ่ายมีเลือดปน','ถ่ายไม่มีเลือดปน','ไม่มีแรง','ร้อนท้อง','แสบท้อง','ผื่น','ผื่นแดง','ผื่นคัน',
           'ชัก','ใจสั่น','เกร็ง','ไทรอยด์','หน้ามืด','Thyrotoxicosis','Thyroid','เป็นลม','หมดสติ','ตาเหลือก','หายใจไม่ออก','thyrotoxicosis','ตุ่มใส','หายใจไม่อิ่ม','เจ็บกลางหน้าอก','บวม','ตุ่ม','ปุ่ม','หัวใจ',
          'ชา','อ่อนแรง','มึนงง','พูดไม่ชัด','ปากเบี้ยว','มึน','เหงื่อ','สั่น','คอบวม','กระวนกระวาย','น้ำหนักลด','หนาว','น้ำหนักขึ้น','อ้วนขึ้น','น้ำหนักเพิ่ม']
custom_words_list = set(thai_words())
custom_words_list.add('ปวดท้องด้านขวาล่าง')
custom_words_list.add('ปวดท้องรอบสะดือ')
custom_words_list.add('เจ็บท้องด้านขวา')
custom_words_list.add('เจ็บท้องข้างขวา')
custom_words_list.add('ปวดท้องด้านขวา')
custom_words_list.add('ปวดท้องข้างขวา')
custom_words_list.add('ปวดท้องตรงกลาง')
custom_words_list.add('ปวดกลางท้อง')
custom_words_list.add('ปวดท้องกลางท้อง')
custom_words_list.add('อุจจาระบ่อย')
custom_words_list.add('น้ำมูกใส')
custom_words_list.add('น้ำมูกข้น')
custom_words_list.add('แผลแห้งดี')
custom_words_list.add('ปวดท้องด้านซ้าย')
custom_words_list.add('ปวดท้องข้างซ้าย')
custom_words_list.add('เจ็บท้องด้านซ้าย')
custom_words_list.add('เจ็บท้องข้างซ้าย')
custom_words_list.add('ปวดท้องน้อย')
custom_words_list.add('ไม่อาเจียน')
custom_words_list.add('แผลที่หน้าท้อง')
custom_words_list.add('ไอแห้ง')
custom_words_list.add('กลืนอาหารลำบาก')
custom_words_list.add('คอแดง')
custom_words_list.add('คัดจมูก')
custom_words_list.add('ผื่น')
custom_words_list.add('คอแห้ง')
custom_words_list.add('หายใจเร็ว')
custom_words_list.add('หอบ')
custom_words_list.add('เจ็บหน้าอก')
custom_words_list.add('แน่นกลางหน้าอก')
custom_words_list.add('เจ็บกลางหน้าอก')
custom_words_list.add('ลิ้นปี่')
custom_words_list.add('ท้องอืด')
custom_words_list.add('ท้องเฟ้อ')
custom_words_list.add('เร่อ')
custom_words_list.add('แน่นท้อง')
custom_words_list.add('เบื่ออาหาร')
custom_words_list.add('แสบหน้าอก')
custom_words_list.add('เร่อเปรี้ยว')
custom_words_list.add('ขมคอ')
custom_words_list.add('ท้องน้อย')
custom_words_list.add('ท้องผูก')
custom_words_list.add('จุกหน้าอก')
custom_words_list.add('ไข้ต่ำ')
custom_words_list.add('ไข้ไม่สูง')
custom_words_list.add('ไข้เล็กน้อย')
custom_words_list.add('ไข้สูง')
custom_words_list.add('เจ็บคอ')
custom_words_list.add('ตัวร้อน')
custom_words_list.add('แสบคอ')
custom_words_list.add('แสบกระเพาะ')
custom_words_list.add('ถ่ายมีเลือดปน')
custom_words_list.add('ถ่ายไม่มีเลือดปน')
custom_words_list.add('ถ่ายบ่อย')
custom_words_list.add('ถ่ายเหลว')
custom_words_list.add('ถ่ายเป็นน้ำ')
custom_words_list.add('จุกหน้าท้อง')
custom_words_list.add('จุกหน้าอก')
custom_words_list.add('จุกอก')
custom_words_list.add('จุกท้อง')
custom_words_list.add('จุกคอ')
custom_words_list.add('แน่นคอ')
custom_words_list.add('เวียนศรีษะ')
custom_words_list.add('ปวดศรีษะ')
custom_words_list.add('ปวดหัว')
custom_words_list.add('เวียนหัว')
custom_words_list.add('ไอปนเลือด')
custom_words_list.add('เสมหะปนเลือด')
custom_words_list.add('ลิ่นปี่')
custom_words_list.add('น้ำมูกเขียว')
custom_words_list.add('น้ำมูกใส')
custom_words_list.add('บริเวณสะดือ')
custom_words_list.add('ไม่มีไข้')
custom_words_list.add('ไม่มีเสมหะ')
custom_words_list.add('ไม่มีน้ำมูก')
custom_words_list.add('ไม่ปวดหัว')
custom_words_list.add('ท้องไม่เสีย')
custom_words_list.add('ไม่มีท้องเสีย')
custom_words_list.add('ถ่ายปกติ')
custom_words_list.add('เพลีย')
custom_words_list.add('ไม่อาเจียน')
custom_words_list.add('ไม่ไอ')
custom_words_list.add('ไม่ปวดท้อง')
custom_words_list.add('ไม่มีแรง')
custom_words_list.add('แสบท้อง')
custom_words_list.add('ร้อนท้อง')
custom_words_list.add('ผื่นแดง')
custom_words_list.add('ผื่นคัน')
custom_words_list.add('แน่นอก')
custom_words_list.add('ตุ่มใส')
custom_words_list.add('หายใจไม่อิ่ม')
custom_words_list.add('อ่อนแรง')
custom_words_list.add('มึนงง')
custom_words_list.add('พูดไม่ชัด')
custom_words_list.add('ปากเบี้ยว')
custom_words_list.add('คอบวม')
custom_words_list.add('กระวนกระวาย')
custom_words_list.add('น้ำหนักลด')
custom_words_list.add('ไทรอยด์')
custom_words_list.add('Thyroid')
custom_words_list.add('thyroid')
custom_words_list.add('แน่นหน้าอกซ้าย')
custom_words_list.add('แน่นหน้าอกข้างซ้าย')
custom_words_list.add('แน่นหน้าอกข้างขวา')
custom_words_list.add('แน่นหน้าอกขวา')
custom_words_list.remove('มีไข้')
custom_words_list.remove('เป็นไข้')
custom_words_list.remove('อาการไอ')

trie = dict_trie(dict_source=custom_words_list)
def tokenize(text):
     tokens = [word for word in word_tokenize(text,custom_dict=trie,keep_whitespace=False)]

     tokens = [item for item in tokens if (item  in symptoms)]
     list2str= ' '.join(tokens)
     return list2str

icd10={'flu':['J9','J10','J11'],
       'common cold':['J00','J31'],
       'pharyngitis':['J02'],
       'bronchitis':['J20','J40','J41','J42'],
       'pneumonia':['J12','J129','J13','J14','J15','J159','J17','J18','J181','J189'],
       'irritable bowel syndrome':['K58'],
       'gastritis':['K29'],
       'appendicitis':['K35','K36','K37'],
       'gastroesophageal reflux':['K21'],
       'ischaemic heart disease':['I20','I21','I22','I23','I24','I25'],
       'rheumatic':['I00','I01','I02','I05','I06','I07','I08','I09'],
       'valve':['I33','I34','I35','I36','I37','I38','I39'],
       'epilepsy':['G40'],
       'encephalitis':['G04','G05','B00.4+','B01.1+'],
       'stroke':['I60','I61','I62','I63','I64','I65','I66','I67','I68','I69'],
       'thyrotoxicosis':['E05'],
       'hypothyroidism':['E03']}

def chest(user_input):
    chest_complaint=[tokenize(user_input)]
    seq = tokenizer_chest.texts_to_sequences(chest_complaint)
    padded = pad_sequences(seq, maxlen=50)
    pred = Chest_model.predict(padded)
    labels=['bronchitis'  ,'common cold', 'flu',  'pharyngitis','pneumonia']
    score=sum(np.array(pred).tolist(), [])
    predict=list(zip(labels, score))
    result = [list(x) for x in predict]
    result=sorted(result,key=lambda x:x[1],reverse=True)
    # create dataframe
    Y=[]
    x=np.arange(0,5)
    for n in range(5):
        y=predict[n][1]*100
        Y.append(y)

    my_xticks=['bronchitis'  ,'common cold', 'flu',  'pharyngitis',' pneumonia']
    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
    df_v=df_v.sort_values('Prob',ascending=False)
    # plot
    plt.figure(figsize=(26,12)).patch.set_facecolor('lavenderblush')
    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(n_colors=5,reverse=True))
    initialx=0
    for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
    plt.yticks(size=22)
    plt.xticks(size=22)
    plt.ylabel('Chest Disease',size=25,labelpad=30,color='black',fontweight="bold")
    plt.xlabel('Probability(%)',size=24,labelpad=15)
    plt.title('Probability Distribution', size=25,y=1.03,color='black',fontweight="bold")
    st.markdown("<font color='indigo' size=5><b>Disease Results:</b></font>",unsafe_allow_html=True)
    st.pyplot()
    st.markdown(" ")
    st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
    for i in range(5):
        st.markdown(f"{result[i][0]} : <font color='green' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)

def gi(user_input):
    GI_complaint=[tokenize(user_input)]
    seq = tokenizer_gi.texts_to_sequences(GI_complaint)
    padded = pad_sequences(seq, maxlen=100)
    pred = GI_model.predict(padded)
    labels=['appendicitis'  ,'gastritis', 'gastroesophageal reflux',  'irritable bowel syndrome']
    score=sum(np.array(pred).tolist(), [])
    predict=list(zip(labels, score))
    result = [list(x) for x in predict]
    result=sorted(result,key=lambda x:x[1],reverse=True)
    # create dataframe
    Y=[]
    x=np.arange(0,4)
    for n in range(4):
        y=predict[n][1]*100
        Y.append(y)

    my_xticks=['appendicitis'  ,'gastritis', 'gastroesophageal reflux',  'irritable bowel syndrome']
    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
    df_v=df_v.sort_values('Prob',ascending=False)
    # plot
    fig = plt.figure(figsize=(40,18),dpi=300)
    fig.patch.set_facecolor('lavenderblush')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.2)
    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(n_colors=4,reverse=True))
    initialx=0
    for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=20)
            initialx+=1
    plt.yticks(size=35)
    plt.xticks(size=32)
    plt.ylabel('GI Disease',size=37,labelpad=30,color='black',fontweight="bold")
    plt.xlabel('Probability(%)',size=37,labelpad=15)
    plt.title('Probability Distribution', size=42,y=1.03,color='black',fontweight="bold")
    st.markdown("<font color='indigo' size=5><b>Disease Results:</b></font>",unsafe_allow_html=True)
    st.pyplot()
    st.markdown(" ")
    st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
    for i in range(4):
        st.markdown(f"{result[i][0]} : <font color='green' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)

def cardio(user_input):
    Cardio_complaint=[tokenize(user_input)]
    seq = tokenizer_cardio.texts_to_sequences(Cardio_complaint)
    padded = pad_sequences(seq, maxlen=100)
    pred = Cardio_model.predict(padded)
    labels=['ischaemic heart disease','rheumatic','valve']
    score=sum(np.array(pred).tolist(), [])
    predict=list(zip(labels, score))
    result = [list(x) for x in predict]
    result=sorted(result,key=lambda x:x[1],reverse=True)
    # create dataframe
    Y=[]
    x=np.arange(0,3)
    for n in range(3):
        y= predict[n][1]*100
        Y.append(y)

    my_xticks=['ischaemic heart disease','rheumatic','valve']
    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
    df_v=df_v.sort_values('Prob',ascending=False)
    # plot
    fig = plt.figure(figsize=(40,18),dpi=300)
    fig.patch.set_facecolor('lavenderblush')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.2)
    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(n_colors=3,reverse=True))
    initialx=0
    for p in splot.patches:
        plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=20)
        initialx+=1
    plt.yticks(size=36)
    plt.xticks(size=30)
    plt.ylabel('Cardio Disease',size=40,labelpad=30,color='black',fontweight="bold")
    plt.xlabel('Probability(%)',size=32,labelpad=15)
    plt.title('Probability Distribution', size=44,y=1.03,color='black',fontweight="bold")
    st.markdown("<font color='indigo' size=5><b>Disease Results:</b></font>",unsafe_allow_html=True)
    st.pyplot()
    st.markdown(" ")
    st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
    for i in range(3):
        st.markdown(f"{result[i][0]} : <font color='green' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)


def neuro(user_input):
    Neuro_complaint=[tokenize(user_input)]
    seq = tokenizer_neuro.texts_to_sequences(Neuro_complaint)
    padded = pad_sequences(seq, maxlen=100)
    pred = Neuro_model.predict(padded)
    labels=['encephalitis','epilepsy','stroke']
    score=sum(np.array(pred).tolist(), [])
    predict=list(zip(labels, score))
    result = [list(x) for x in predict]
    result=sorted(result,key=lambda x:x[1],reverse=True)
    # create dataframe
    Y=[]
    x=np.arange(0,3)
    for n in range(3):
        y=predict[n][1]*100
        Y.append(y)

    my_xticks=['encephalitis','epilepsy','stroke']
    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
    df_v=df_v.sort_values('Prob',ascending=False)
    # plot
    fig = plt.figure(figsize=(30,16),dpi=300)
    fig.patch.set_facecolor('lavenderblush')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.15)
    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(n_colors=3,reverse=True))
    initialx=0
    for p in splot.patches:
        plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=18)
        initialx+=1
    plt.yticks(size=28)
    plt.xticks(size=24)
    plt.ylabel('Neuro Disease',size=34,labelpad=30,color='black',fontweight="bold")
    plt.xlabel('Probability(%)',size=28,labelpad=15)
    plt.title('Probability Distribution', size=38,y=1.03,color='black',fontweight="bold")
    st.markdown("<font color='indigo' size=5><b>Disease Results:</b></font>",unsafe_allow_html=True)
    st.pyplot()
    st.markdown(" ")
    st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
    for i in range(3):
        st.markdown(f"{result[i][0]} : <font color='green' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)

def endocrine(user_input):
    Endocrine_complaint=[tokenize(user_input)]
    seq = tokenizer_endocrine.texts_to_sequences(Endocrine_complaint)
    padded = pad_sequences(seq, maxlen=100)
    pred = Endocrine_model.predict(padded)
    labels=['hypothyroidism','thyrotoxicosis']
    score=sum(np.array(pred).tolist(), [])
    predict=list(zip(labels, score))
    result = [list(x) for x in predict]
    result=sorted(result,key=lambda x:x[1],reverse=True)
    # create dataframe
    Y=[]
    x=np.arange(0,2)
    for n in range(2):
        y=predict[n][1]*100
        Y.append(y)

    my_xticks=['hypothyroidism','thyrotoxicosis']
    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
    df_v=df_v.sort_values('Prob',ascending=False)
    # plot
    fig = plt.figure(figsize=(30,17),dpi=300)
    fig.patch.set_facecolor('lavenderblush')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.2)
    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(n_colors=2,reverse=True))
    initialx=0
    for p in splot.patches:
        plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=15)
        initialx+=1
    plt.yticks(size=32)
    plt.xticks(size=26)
    plt.ylabel('Endocrine Disease',size=34,labelpad=30,color='black',fontweight="bold")
    plt.xlabel('Probability(%)',size=32,labelpad=15)
    plt.title('Probability Distribution', size=38,y=1.03,color='black',fontweight="bold")
    st.markdown("<font color='indigo' size=5><b>Disease Results:</b></font>",unsafe_allow_html=True)
    st.pyplot()
    st.markdown(" ")
    st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
    for i in range(2):
        st.markdown(f"{result[i][0]} : <font color='green' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)


#st.set_page_config(layout="wide")

def app():

    # st.markdown(""" <style> .font {
    # font-size:38px ; font-family: 'fantasy'; color: #ffffff; background-image: linear-gradient(to right,darkslategray,darkslategray); text-align:center}
    # </style> """, unsafe_allow_html=True)
    # st.markdown('<p class="font"><b>Diagnosis Prediction based on symptoms</b></p>', unsafe_allow_html=True)
    st.markdown("<font color='grey' size=4 ><b>17 diseases from 5 groups as follows :</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey'><b>Chest : flu, common cold, pharyngitis, bronchitis and pneumonia</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey'><b>GI : irritable bowel syndrome, gastritis, appendicitis and gastroesophageal reflux</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey'><b>Cardio : ischaemic heart disease, rheumatic and valve</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey'><b>Neuro: epilepsy, encephalitis and stroke</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey'><b>Endocrine : thyrotoxicosis and hypothyroidism</b></font>",unsafe_allow_html=True)
    st.markdown(' ')

    input_type = st.radio(
         "Please select a input type.",
         ('Text', 'Voice'))

    if input_type =='Text':
        user_input = st.text_area("Please fill in your symptoms.", height=90)

        st.markdown("<font color='black' size=2>Example : ไอแห้งๆ หอบเหนื่อย รู้สึกแน่นหน้าอก</font>",unsafe_allow_html=True)


        m = st.markdown("""<style>div.stButton > button:first-child {
            background-color: limegreen;
          border-color: limegreen;
          color: white;
          padding: 6px 12px;
          text-decoration: none;
          margin: 2px 1px;
          cursor: pointer;
          border-radius: 10px;}
            </style>""", unsafe_allow_html=True)

        if st.button('predict'):
            if user_input =='':
                st.markdown("<font color='darkred' size=5><b>Sorry, please fill in your symptoms.</b></font>",unsafe_allow_html=True)

            if user_input != '':
                new_complaint=[tokenize(user_input)]
                seq = tokenizer_group.texts_to_sequences(new_complaint)
                padded = pad_sequences(seq, maxlen=100)
                pred = Group_model.predict(padded)
                labels=['Cardio','Chest','Endocrine','GI','Neuro']
                score=sum(np.array(pred).tolist(), [])
                score =[round(num, 4) for num in score]
                predict=list(zip(labels, score))
                result = [list(x) for x in predict]
                result=sorted(result,key=lambda x:x[1],reverse=True)
                if result[0][1]==0.2655 and result[1][1]==0.2377 and result[2][1]==0.1946 and result[3][1]==0.1553 and result[4][1]==0.1469 :
                    st.markdown("<font color='darkred' size=5><b>Sorry, please try other words.</b></font>",unsafe_allow_html=True)
                else:
                    # create dataframe
                    Y=[]
                    x=np.arange(0,5)
                    for n in range(5):
                        y=predict[n][1]*100
                        Y.append(y)

                    my_xticks=['Cardio','Chest','Endocrine','GI','Neuro']
                    df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
                    df_v=df_v.sort_values('Prob',ascending=False)



                    # plot
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.set_style("whitegrid")
                    plt.figure(figsize=(22,12)).patch.set_facecolor('aliceblue')
                    splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=5))
                    initialx=0
                    for p in splot.patches:
                            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
                            initialx+=1
                    plt.yticks(size=24)
                    plt.xticks(size=22)
                    plt.ylabel('Group',size=25,labelpad=30,color='black',fontweight="bold")
                    plt.xlabel('Probability(%)',size=22,labelpad=15)
                    plt.title('Probability Distribution', size=30,y=1.03,color='black',fontweight="bold")
                    st.markdown("<font color='mediumblue' size=5><b>Group Results:</b></font>",unsafe_allow_html=True)
                    st.pyplot()
                    if result[0][0] =='Chest':
                        chest(user_input)

                    if result[0][0] =='GI':
                        gi(user_input)

                    if result[0][0] =='Cardio':
                        cardio(user_input)

                    if result[0][0] =='Neuro':
                        neuro(user_input)

                    if result[0][0] =='Endocrine':
                        endocrine(user_input)


    if input_type =='Voice':
        #st.markdown("<font color='darkblue' size=4>you can use recording function, just click button below.</font>",unsafe_allow_html=True)
        stt_button = Button(label="Speak", width=100,button_type="primary")

        stt_button.js_on_event("button_click",CustomJS(code="""alert("Click 'ok' to start recording your voice.")"""))
        stt_button.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang ='th-TH';
            
        
            recognition.onresult = function (e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                if ( value != "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                    recognition.stop()
                }
            }
            recognition.start();
            """))

        result = streamlit_bokeh_events(
                stt_button,
                events="GET_TEXT",
                key="listen",
                refresh_on_update=True,
                override_height=75,
                debounce_time=0)

        if result:
            if "GET_TEXT" in result:
                user_input=result.get('GET_TEXT')
                st.markdown(f"<font color='black' size=4 ><b>บันทึกอาการของคุณ : </b></font><font color='darkblue' size=4 ><b>{result.get('GET_TEXT')}</b></font>",unsafe_allow_html=True)
                m = st.markdown("""<style>div.stButton > button:first-child {
                background-color: limegreen;
                border-color: limegreen;
                color: white;
                padding: 6px 12px;
                text-decoration: none;
                margin: 2px 1px;
                cursor: pointer;
                border-radius: 10px;}
               </style>""", unsafe_allow_html=True)
                if st.button('predict'):
                    if user_input =='':
                        st.markdown("<font color='darkred' size=5><b>Sorry, please fill in your symptoms.</b></font>",unsafe_allow_html=True)

                    if user_input != '':
                        new_complaint=[tokenize(user_input)]
                        seq = tokenizer_group.texts_to_sequences(new_complaint)
                        padded = pad_sequences(seq, maxlen=100)
                        pred = Group_model.predict(padded)
                        labels=['Cardio','Chest','Endocrine','GI','Neuro']
                        score=sum(np.array(pred).tolist(), [])
                        score =[round(num, 4) for num in score]
                        predict=list(zip(labels, score))
                        result = [list(x) for x in predict]
                        result=sorted(result,key=lambda x:x[1],reverse=True)
                        if result[0][1]==0.2655 and result[1][1]==0.2377 and result[2][1]==0.1946 and result[3][1]==0.1553 and result[4][1]==0.1469 :
                            st.markdown("<font color='darkred' size=5><b>Sorry, please try other words.</b></font>",unsafe_allow_html=True)
                        else:
                            # create dataframe
                            Y=[]
                            x=np.arange(0,5)
                            for n in range(5):
                                y=predict[n][1]*100
                                Y.append(y)

                            my_xticks=['Cardio','Chest','Endocrine','GI','Neuro']
                            df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
                            df_v=df_v.sort_values('Prob',ascending=False)



                            # plot
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            sns.set_style("whitegrid")
                            plt.figure(figsize=(22,12)).patch.set_facecolor('aliceblue')
                            splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=5))
                            initialx=0
                            for p in splot.patches:
                                    plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
                                    initialx+=1
                            plt.yticks(size=24)
                            plt.xticks(size=22)
                            plt.ylabel('Group',size=25,labelpad=30,color='black',fontweight="bold")
                            plt.xlabel('Probability(%)',size=22,labelpad=15)
                            plt.title('Probability Distribution', size=30,y=1.03,color='black',fontweight="bold")
                            st.markdown("<font color='mediumblue' size=5><b>Group Results:</b></font>",unsafe_allow_html=True)
                            st.pyplot()
                            if result[0][0] =='Chest':
                                chest(user_input)

                            if result[0][0] =='GI':
                                gi(user_input)

                            if result[0][0] =='Cardio':
                                cardio(user_input)

                            if result[0][0] =='Neuro':
                                neuro(user_input)

                            if result[0][0] =='Endocrine':
                                endocrine(user_input)

