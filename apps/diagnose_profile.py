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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df= pd.read_csv('new_model/for_dummy.csv')
df= df.drop(['Unnamed: 0'],axis=1)


Group_model = load_model('new_model/Group_model.h5')
GI_model = load_model('new_model/GI_model.h5')
Cardio_model = load_model('new_model/Cardio_model.h5')
Chest_model = load_model('new_model/Chest_model.h5')
Neuro_model = load_model('new_model/Neuro_model.h5')
Endocrine_model = load_model('new_model/Endocrine_model.h5')
# load vectorizer.vocabulary_
import pickle
tokenizer_group = pickle.load(open("new_model/Group_tokenizer.pickle", 'rb'))
tokenizer_chest = pickle.load(open("new_model/Chest_tokenizer.pickle", 'rb'))
tokenizer_gi = pickle.load(open("new_model/GI_tokenizer.pickle", 'rb'))
tokenizer_cardio = pickle.load(open("new_model/Cardio_tokenizer.pickle", 'rb'))
tokenizer_neuro = pickle.load(open("new_model/Neuro_tokenizer.pickle", 'rb'))
tokenizer_endocrine = pickle.load(open("new_model/Endocrine_tokenizer.pickle", 'rb'))

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

def chest_prediction(symptom,disease,temp):
    test_df=df.copy()
    test_df=test_df[['UnderlyingDisease']]
    test_df.loc[len(test_df)]=[disease]
    dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
    dummies=dummies1.to_numpy()
    dummies=[dummies[-1]]

    Disease=np.array(dummies)

    new_complaint=[tokenize(symptom)]
    Temperature=np.array([temp])

    seq = tokenizer_chest.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=15)

    pred = Chest_model.predict([padded,Disease,Temperature] ,verbose=0)
    labels=['common cold','bronchitis', 'pharyngitis','pneumonia','flu']
    if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
         st.markdown('Sorry, please try other words.')
    else:
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

        my_xticks=['common cold','bronchitis', 'pharyngitis','pneumonia','flu']
        df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
        df_v=df_v.sort_values('Prob',ascending=False)
        # plot

        plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=5))
        initialx=0
        for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
        plt.yticks(size=20)
        plt.xticks(size=18)
        plt.ylabel('Chest',size=22,labelpad=30,color='navy',fontweight="bold")
        plt.xlabel('Probability',size=15,labelpad=15)
        plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
        st.markdown("<font color='black' size=4><b>Disease Result:</b></font>",unsafe_allow_html=True)
        st.pyplot()
        st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
        for i in range(5):
            st.markdown(f"{result[i][0]} : <font color='mediumblue' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)
def gi_prediction(symptom,disease,bw,temp,bmi,age,sex,exercise,smoke,narcotic):
    test_df=df.copy()
    test_df=test_df[['UnderlyingDisease']]
    test_df.loc[len(test_df)]=[disease]
    dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
    dummies=dummies1.to_numpy()
    dummies=[dummies[-1]]

    Disease=np.array(dummies)

    new_complaint=[tokenize(symptom)]
    Bw=np.array([bw])
    Temperature=np.array([temp])
    Bmi=np.array([bmi])
    Age=np.array([age])
    Sex=np.array([sex])
    Exer=np.array([exercise])
    Smoke=np.array([smoke])
    Narcotic=np.array([narcotic])


    seq = tokenizer_gi.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=50)

    pred = GI_model.predict([padded,Disease,Bw,Temperature,Bmi,Age,Sex,Exer,Smoke,Narcotic] ,verbose=0)
    labels=['gastritis', 'gastroesophageal reflux','appendicitis',  'irritable bowel syndrome']
    if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
        st.markdown('Sorry, please try other words.')
    else:

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

        my_xticks=['gastritis', 'gastroesophageal reflux','appendicitis',  'irritable bowel syndrome']
        df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
        df_v=df_v.sort_values('Prob',ascending=False)
        # plot

        plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=4))
        initialx=0
        for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
        plt.yticks(size=20)
        plt.xticks(size=18)
        plt.ylabel('GI',size=22,labelpad=30,color='navy',fontweight="bold")
        plt.xlabel('Probability',size=15,labelpad=15)
        plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
        st.markdown("<font color='black' size=4><b>Disease Result:</b></font>",unsafe_allow_html=True)
        st.pyplot()
        st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
        for i in range(4):
            st.markdown(f"{result[i][0]} : <font color='mediumblue' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)
def cardio_prediction(symptom,disease,age):
    test_df=df.copy()
    test_df=test_df[['UnderlyingDisease']]
    test_df.loc[len(test_df)]=[disease]
    dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
    dummies=dummies1.to_numpy()
    dummies=[dummies[-1]]

    Disease=np.array(dummies)

    new_complaint=[tokenize(symptom)]
    Age=np.array([age])

    seq = tokenizer_cardio.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=50)

    pred = Cardio_model.predict([padded,Disease,Age] ,verbose=0)
    labels=['ischaemic heart disease','valve','rheumatic']
    if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
        st.markdown('Sorry, please try other words.')
    else:
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

        my_xticks=['ischaemic heart disease','valve','rheumatic']
        df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
        df_v=df_v.sort_values('Prob',ascending=False)
        # plot

        plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=3))
        initialx=0
        for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
        plt.yticks(size=20)
        plt.xticks(size=18)
        plt.ylabel('Cardio',size=22,labelpad=30,color='navy',fontweight="bold")
        plt.xlabel('Probability',size=15,labelpad=15)
        plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
        st.markdown("<font color='black' size=4><b>Disease Result:</b></font>",unsafe_allow_html=True)
        st.pyplot()
        st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
        for i in range(3):
            st.markdown(f"{result[i][0]} : <font color='mediumblue' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)
def neuro_prediction(symptom,disease,bpd,bps,exercise,pulse,temp,rr,hr,waist,age,narcotic,sex):
    test_df=df.copy()
    test_df=test_df[['UnderlyingDisease']]
    test_df.loc[len(test_df)]=[disease]
    dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
    dummies=dummies1.to_numpy()
    dummies=[dummies[-1]]

    Disease=np.array(dummies)

    new_complaint=[tokenize(symptom)]
    Bpd=np.array([bpd])
    Bps=np.array([bps])
    Exer=np.array([exercise])
    Pulse=np.array([pulse])
    Temperature=np.array([temp])
    RR=np.array([rr])
    Hr=np.array([hr])
    Waist=np.array([waist])
    Age=np.array([age])
    Narcotic=np.array([narcotic])
    Sex=np.array([sex])

    seq = tokenizer_neuro.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=50)

    pred = Neuro_model.predict([padded,Disease,Bpd,Bps,Exer,Pulse,Temperature,RR,Hr,Waist,Age,Narcotic,Sex] ,verbose=0)
    labels=['stroke','epilepsy','encephalitis']
    if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
        st.markdown('Sorry, please try other words.')
    else:
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

        my_xticks=['stroke','epilepsy','encephalitis']
        df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
        df_v=df_v.sort_values('Prob',ascending=False)
        # plot

        plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=3))
        initialx=0
        for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
        plt.yticks(size=20)
        plt.xticks(size=18)
        plt.ylabel('Neuro',size=22,labelpad=30,color='navy',fontweight="bold")
        plt.xlabel('Probability',size=15,labelpad=15)
        plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
        st.markdown("<font color='black' size=4><b>Disease Result:</b></font>",unsafe_allow_html=True)
        st.pyplot()
        st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
        for i in range(3):
            st.markdown(f"{result[i][0]} : <font color='mediumblue' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)
def endocrine_prediction(symptom,disease,pulse):
    test_df=df.copy()
    test_df=test_df[['UnderlyingDisease']]
    test_df.loc[len(test_df)]=[disease]
    dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
    dummies=dummies1.to_numpy()
    dummies=[dummies[-1]]

    Disease=np.array(dummies)

    new_complaint=[tokenize(symptom)]
    Pulse=np.array([pulse])

    seq = tokenizer_endocrine.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=50)

    pred = Endocrine_model.predict([padded,Disease,Pulse] ,verbose=0)
    labels=['thyrotoxicosis','hypothyroidism']
    if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
        st.markdown('Sorry, please try other words.')
    else:

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

        my_xticks=['thyrotoxicosis','hypothyroidism']
        df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
        df_v=df_v.sort_values('Prob',ascending=False)
        # plot

        plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=2))
        initialx=0
        for p in splot.patches:
            plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
            initialx+=1
        plt.yticks(size=20)
        plt.xticks(size=18)
        plt.ylabel('Endocrine',size=22,labelpad=30,color='navy',fontweight="bold")
        plt.xlabel('Probability',size=15,labelpad=15)
        plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
        st.markdown("<font color='black' size=4><b>Disease Result:</b></font>",unsafe_allow_html=True)
        st.pyplot()
        st.markdown("<font color='black' size=4><b><u>ICD 10 codes:</u></b></font>",unsafe_allow_html=True)
        for i in range(2):
            st.markdown(f"{result[i][0]} : <font color='mediumblue' size=3><b>{','.join(icd10.get(result[i][0]))}</b></font>",unsafe_allow_html=True)



#features = pd.DataFrame(data, index=[0])
# Displays the user input features

def app():
    user_input = st.sidebar.text_area("Please fill your symptoms.", height=90)
    yes_no = st.sidebar.selectbox('Do you have any underlying disease?', ('no', 'yes'))
    if yes_no =='yes':
        options = st.sidebar.multiselect('What underlying disease do you have?',['โรคไตเรื้อรัง', 'โรคไขมันในเลือดสูง', 'โรคเบาหวาน', 'โรคอ้วน','โรคหืด','โรคหัวใจขาดเลือด','โรคหลอดเลือดสมอง','โรคลมชัก','โรคธาลัสซีเมีย','โรคถุงลมโป่งพอง','โรคความดันโลหิตสูง','เนื้องอกสมอง','มะเร็งเม็ดเลือดขาว','มะเร็งเต้านม','มะเร็งลำไส้ใหญ่','มะเร็งปากมดลูก','มะเร็งปอด','มะเร็งต่อมไทรอยด์','มะเร็งตับ','ภาวะไทรอยด์เป็นพิษ','ภาวะไทรอยด์ต่ำ','ภาวะซีด','นิ่วในถุงน้ำดี'])
    else:
        options=['ไม่มีโรคประจำตัว']
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    age = st.sidebar.slider('Age', 0, 120, 40)
    temp=st.sidebar.slider('Temperature',33.00,44.00,36.5)
    bps = st.sidebar.slider('BPS', 30, 240, 120)
    bpd = st.sidebar.slider('BPD', 30, 180, 80)
    bw = st.sidebar.slider('Weight', 30.0, 180.0, 60.0)
    bt = st.sidebar.slider('Height', 30, 200, 160)
    waist = st.sidebar.slider('Waist', 30.0, 180.0, 75.0)
    rr = st.sidebar.slider('Respiratory rate', 5, 50, 15)
    hr = st.sidebar.slider('Heart rate', 10, 200, 75)
    pulse = st.sidebar.slider('Pulse', 30, 180, 75)
    exercise = st.sidebar.selectbox('Exercise', ('no', 'yes'))
    smoke = st.sidebar.selectbox('Smoking', ('no', 'yes'))
    narcotic = st.sidebar.selectbox('Narcotic', ('no', 'yes'))
    Sex=[]
    if sex =='male':
        Sex.append(1)
    else:
        Sex.append(0)
    Exercise=[]
    if exercise =='yes':
        Exercise.append(1)
    else:
        Exercise.append(0)
    Smoke=[]
    if smoke =='yes':
        Smoke.append(1)
    else:
        Smoke.append(0)
    Narcotic=[]
    if narcotic =='yes':
        Narcotic.append(1)
    else:
        Narcotic.append(0)
    bt = float(bt) / 100
    bw = float(bw)
    bmi = round(bw / (bt * bt), 2)

    Options = '  '.join(options)
    Options=Options.replace("มะเร็งต่อมไทรอยด์","มะเร็งต่อไทรอยด์")


    data = [user_input,Options,Sex[0],age,temp,bpd,bps,bw,bt,bmi,waist,rr,hr,pulse,Exercise[0],Smoke[0],Narcotic[0]]

    # st.markdown("<font color='navy' size=6 ><b>Diagnosis Prediction based on Clinical Text</b></font>" ,unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3 ><b>17 diseases from 5 groups as follows :</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3><b>Chest : flu, common cold, pharyngitis, bronchitis and pneumonia</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3><b>GI : irritable bowel syndrome, gastritis, appendicitis and gastroesophageal reflux</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3><b>Cardio : ischaemic heart disease, rheumatic and valve</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3><b>Neuro: epilepsy, encephalitis and stroke</b></font>",unsafe_allow_html=True)
    st.markdown("<font color='grey' size=3><b>Endocrine : thyrotoxicosis and hypothyroidism</b></font>",unsafe_allow_html=True)
    if user_input != '':
        test_df=df.copy()
        test_df=test_df[['UnderlyingDisease']]
        test_df.loc[len(test_df)]=[Options]
        dummies1=test_df['UnderlyingDisease'].str.get_dummies(sep=' ')
        dummies=dummies1.to_numpy()
        dummies=[dummies[-1]]

        Disease=np.array(dummies)

        new_complaint=[tokenize(user_input)]
        Temperature=np.array([temp])
        Age=np.array([age])
        gender=np.array([Sex[0]])
        seq = tokenizer_group.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=15)

        pred = Group_model.predict([padded,Disease,Temperature,Age,gender] ,verbose=0)
        labels=['Cardio','Chest','Endocrine','GI','Neuro']
        if ((padded==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all()) and ((Disease==np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all()):
            st.markdown("<font color ='red' size=5><b>Sorry, please try other words.</b></font>",unsafe_allow_html=True)
        else:
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

            my_xticks=['Cardio','Chest','Endocrine','GI','Neuro']
            df_v=pd.DataFrame({'Disease': my_xticks,'Prob':Y})
            df_v=df_v.sort_values('Prob',ascending=False)
            # plot
            sns.set_style("whitegrid")
            plt.figure(figsize=(20,10)).patch.set_facecolor('aliceblue')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            splot=sns.barplot(x="Prob",y="Disease",data=df_v,palette=sns.cubehelix_palette(rot=-.2,reverse=True,n_colors=5))
            initialx=0
            for p in splot.patches:
                plt.text(p.get_width()+0.001,initialx+p.get_height()/8,'  '+str(round(p.get_width(),2))+'%',size=12)
                initialx+=1
            plt.yticks(size=20)
            plt.xticks(size=18)
            plt.ylabel('Group',size=22,labelpad=30,color='navy',fontweight="bold")
            plt.xlabel('Probability',size=15,labelpad=15)
            plt.title('Probability Distribution', size=22,y=1.03,color='navy',fontweight="bold")
            st.markdown("<font color='black' size=4><b>Group Result:</b></font>",unsafe_allow_html=True)
            st.pyplot()
            if result[0][0] =='Chest':
                chest_prediction(user_input,Options,temp)
            if result[0][0] == 'GI':
                gi_prediction(user_input,Options,bw,temp,bmi,age,Sex[0],Exercise[0],Smoke[0],Narcotic[0])
            if result[0][0] == 'Cardio':
                cardio_prediction(user_input,Options,age)
            if result[0][0] == 'Neuro' :
                neuro_prediction(user_input,Options,bpd,bps,Exercise[0],pulse,temp,rr,hr,waist,age,Narcotic[0],Sex[0])
            if result[0][0] == 'Endocrine':
                endocrine_prediction(user_input,Options,pulse)
