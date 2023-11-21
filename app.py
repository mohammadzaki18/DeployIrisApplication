import streamlit as st
import sys
sys.path.insert(0, './predict')
from predict import prediction_newdata # replace with your actual function name

st.text("Latihan Machine Learning")
st.title('Iris Feature Input App')

# Fungsi untuk menyimpan fitur ke CSV
def save_features_to_csv(features):
    # Membaca fitur yang sudah ada (jika file ada)
    try:
        existing_data = pd.read_csv('iris_features.csv')
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])

    # Menambahkan data baru
    new_data = pd.DataFrame([features], columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
    updated_data = existing_data.append(new_data, ignore_index=True)

    # Menyimpan ke CSV
    updated_data.to_csv('iris_features.csv', index=False)

# Membuat input untuk setiap fitur
sepal_length = st.number_input('Sepal Length', value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width', value=0.0, step=0.1)
petal_length = st.number_input('Petal Length', value=0.0, step=0.1)
petal_width = st.number_input('Petal Width', value=0.0, step=0.1)

if st.button('Prediksi'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    y_pred = prediction_newdata([features])
    st.write('Prediction as : ', y_pred)