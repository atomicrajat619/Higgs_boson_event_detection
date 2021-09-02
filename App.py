import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,normalize
from tensorflow.keras.layers import Conv2D,LSTM,LeakyReLU, MaxPooling2D,Concatenate,Input, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model

model = tf.keras.models.load_model('higgs.h5')

def prediction(model,input):
    prediction = model.predict(input)
    print('Prediction Completed')
    return 's' if prediction[0][0] >= 0.5 else 'b'

def proba(model,input):
    proba = model.predict(input)
    print('Probability of predicted class')
    return proba

col = [['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
       'DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta',
       'PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi','PRI_met',
       'PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta',
       'PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt']]

def main():
    st.header('Higgs Boson Event Detection Projectt')

    st.write('It demonstrates Deployment of Higgs Boson event detection project and displaying its results')
    st.write('The model was trained on the Higgs Boson dataset')

    st.subheader('Input the Data')
    st.write('Please input the data below')

    a = st.number_input('DER_mass_MMC',)
    b = st.number_input('DER_mass_transverse_met_lep',)
    c = st.number_input('DER_mass_vis',)
    d = st.number_input('DER_pt_h',)
    e = st.number_input('DER_deltaeta_jet_jet',)
    f = st.number_input('DER_mass_jet_jet',)
    g = st.number_input('DER_prodeta_jet_jet',)
    h = st.number_input('DER_deltar_tau_lep',)
    i = st.number_input('DER_pt_tot',)
    j = st.number_input('DER_sum_pt',)
    k = st.number_input('DER_pt_ratio_lep_tau',)
    l = st.number_input('DER_met_phi_centrality',)
    m = st.number_input('DER_lep_eta_centrality',)
    n = st.number_input('PRI_tau_pt',)
    o = st.number_input('PRI_tau_eta',)
    p = st.number_input('PRI_tau_phi',)
    q = st.number_input('PRI_lep_pt',)
    r = st.number_input('PRI_lep_eta',)
    s = st.number_input('PRI_lep_phi',)
    t = st.number_input('PRI_met',)
    u = st.number_input('PRI_met_phi',)
    v = st.number_input('PRI_met_sumet',)
    w = st.number_input('PRI_jet_num',)
    x = st.number_input('PRI_jet_leading_pt',)
    y = st.number_input('PRI_jet_leading_eta',)
    z = st.number_input('PRI_jet_leading_phi',)
    A = st.number_input('PRI_jet_subleading_pt',)
    B = st.number_input('PRI_jet_subleading_eta',)
    C = st.number_input('PRI_jet_subleading_phi',)
    D = st.number_input('PRI_jet_all_pt',)



    input = np.array([[i,j,k,l,m,n,o]])
    print(type(i))
    print(input)
    
    
    if st.button('Detect Event'):
        pred = prediction(model,input)        
        st.success('The event is predicted is ' + pred)

    if st.button('Show Probability'):
        prob = proba(model,input)
        st.success('The probability of the event is {}'.format(prob[0][0]))

if __name__ == '__main__':

    main()