#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xgboost
import streamlit as st
import joblib
import numpy as np
import pandas as pd


# In[4]:


# 加载模型
model = joblib.load('XGBoost_3.9.pkl')  # 确保路径正确

# Streamlit用户界面
st.title("Clincoradiomic Prediction Model for SDB")

# Liver Cirrhosis 输入
liver_cirrhosis = st.selectbox("Liver Cirrhosis (0=Absent, 1=Present):", options=[0, 1], format_func=lambda x: 'Absent (0)' if x == 0 else 'Present (1)')

# 其他输入
dor_dot = st.number_input("DoR/DoT:", min_value=-1e308, max_value=1e308, value=-0.0055)
Radiomic_feature_1 = st.number_input("AP_wavelet_HHL_ngtdm_Coarseness:", min_value=-1e308, max_value=1e308, value=-0.0055)
Radiomic_feature_2 = st.number_input("AP_wavelet_LLL_glcm_Imc1:", min_value=-1e308, max_value=1e308, value=-0.0055)
Radiomic_feature_3 = st.number_input("AP_wavelet_HLL_glszm_GrayLevelNonUniformity:", min_value=-1e308, max_value=1e308, value=-0.0055)
Radiomic_feature_4 = st.number_input("PP_original_shape_Sphericity:", min_value=-1e308, max_value=1e308, value=-0.0055)

# 特征归一化计算
AP_wavelet_HHL_ngtdm_Coarseness = (Radiomic_feature_1 - 0.697121615332527) / 0.0797369413455108
AP_wavelet_LLL_glcm_Imc1 = (Radiomic_feature_2 + 0.238933160020595) / 0.16810763993149
AP_wavelet_HLL_glszm_GrayLevelNonUniformity = (Radiomic_feature_3 - 10.7075705269179) / 14.2559525195507
PP_original_shape_Sphericity = (Radiomic_feature_4 - 0.697121615332527) / 0.0797369413455108

# 计算 rad_score
rad_score = 0.5243 + 0.0769 * PP_original_shape_Sphericity + 0.0022 * AP_wavelet_HHL_ngtdm_Coarseness + 0.0706 * AP_wavelet_LLL_glcm_Imc1 - 0.0286 * AP_wavelet_HLL_glszm_GrayLevelNonUniformity

# 将 dor_dot 放大
dor_dot = dor_dot * 10000

# 构建模型输入特征数组
features = np.array([[rad_score, dor_dot, liver_cirrhosis]])

# 预测按钮
if st.button("PREDICT"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # 根据预测类别显示不同的结果
    if predicted_class == 0:
        result_text = "SD Non Benefiting (SDNB)"
    else:
        result_text = "SD Benefiting (SDB)"
    
    # 显示预测结果
    st.write(f"**Predict Class:** {result_text}")
    st.write(f"**Predict Probabilities:** {predicted_proba}")


# In[ ]:





# In[ ]:




