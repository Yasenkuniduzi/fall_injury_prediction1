import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# 加载模型
model_path = "/Volumes/mac/CHARLS/2015_2018_2020data/gradient_boosting_fall_injury_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# 定义预测函数
def predict_fall_injury(input_data):
    """
    输入患者特征，输出预测结果。
    参数:
        input_data (dict): 包含患者特征的字典
    返回:
        预测结果 (概率) 和分类
    """
    input_df = pd.DataFrame([input_data])
    required_features = ['Age', 'BMI', 'Hypertension', 'Diabetes', 'ChronicDiseaseCount', 'HistoryOfFalls']
    input_df = input_df[required_features]
    prediction_prob = model.predict_proba(input_df)[:, 1][0]
    prediction_class = model.predict(input_df)[0]
    return prediction_prob, prediction_class

# Streamlit 应用界面
# 页面标题和LOGO
st.image("/Volumes/mac/博士/襄阳中心医院logol.jpeg", width=150)  # LOGO 图像路径
st.title("襄阳中心医院跌倒相关损伤预测系统")

# 输入表单
age = st.number_input("年龄", min_value=0, max_value=120, step=1, value=65)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=25.0)
hypertension = st.selectbox("高血压 (0: 无, 1: 有)", [0, 1])
diabetes = st.selectbox("糖尿病 (0: 无, 1: 有)", [0, 1])
chronic_diseases = st.number_input("慢性病数量", min_value=0, max_value=10, step=1, value=2)
history_of_falls = st.selectbox("跌倒史 (0: 无, 1: 有)", [0, 1])

# 预测按钮
if st.button("预测"):
    # 组装输入数据
    patient_data = {
        'Age': age,
        'BMI': bmi,
        'Hypertension': hypertension,
        'Diabetes': diabetes,
        'ChronicDiseaseCount': chronic_diseases,
        'HistoryOfFalls': history_of_falls
    }
    
    # 获取预测结果
    probability, prediction = predict_fall_injury(patient_data)
    st.subheader("预测结果")
    st.write(f"预测跌倒相关损伤概率: {probability:.2f}")
    st.write(f"分类结果: {'有风险' if prediction == 1 else '无风险'}")
    
    # 显示饼图
    labels = ['No Risk', 'Risk']
    sizes = [1 - probability, probability]
    colors = ['lightblue', 'salmon']
    explode = (0, 0.1)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title('Fall-Related Injury Risk Prediction')
    st.pyplot(fig)
