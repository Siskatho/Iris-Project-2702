import streamlit as st
import pandas as pd
import joblib

#How to run streamlit at local
#open terminal, run 'streamlit_app.py'


st.title("Iris Classifier")  #buat judul aplikasi
st.write("This is a simple Iris Classifier app")  #untuk tampilkan text seperti print

# Inference Function
# Dibuat setelah judul dan sebelum user input
model = joblib.load("model_new.joblib")

def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

#untuk user input
left, right = st.columns(2, gap="medium")

# -- Sepal Input
left.subheader("Sepal")
sepal_length = left.slider("Sepal Length", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
sepal_width = left.slider("Sepal Width", min_value=1.0, max_value=10.1, value=5.4, step=0.1)


# -- Petal Input
right.subheader("Petal")
petal_length = right.slider("Petal Length", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
petal_width = right.slider("Petal Width", min_value=1.0, max_value=10.1, value=5.4, step=0.1)


#Show Input Value
#st.dataframe utk tampilkan dalam bentuk data frame
# use_container_width=True berfungsi utk mentatur lebar dataframe,
# kalau true jadinya lebarnya data frame sesuai dengan data container

data = pd.DataFrame({"sepal length (cm)": [sepal_length],
                    "sepal width (cm)": [sepal_width],
                    "petal length (cm)": [petal_length],
                    "petal width (cm)": [petal_width]})

st.dataframe(data,use_container_width=True)

#Prediction Button
button = st.button("Predict", use_container_width=True) 

if button:
    st.write("Prediksi Berhasil!")
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    output = f"Iris anda di classifikasikan sebagai {label_proba:.0%} {label_pred}"
    st.write(output)




