import joblib
import numpy as np


def prediction_newdata(newData):
    filename = './model/iris_svm_model.sav'
    loaded_model = joblib.load(filename)

    # Memuat model yang telah disimpan dari file
    filename = './model/iris_svm_model.sav'
    loaded_model = joblib.load(filename)

    # Memuat scaler yang telah disimpan dari file
    scaler_filename = './model/scaler.save'
    scaler = joblib.load(scaler_filename)

    # Jangan lupa untuk mengubah fitur data baru dengan scaler yang sama yang digunakan saat pelatihan
    new_data_scaled = scaler.transform(newData)

    # Membuat prediksi menggunakan model yang telah dilatih
    predictions = loaded_model.predict(new_data_scaled)
    result = ""
    if predictions[0] == 0:
        result = 'Setosa'
    elif predictions[0] == 1:
        result = 'Versicolor'
    else:
        result = 'Virginica'
    # Mencetak prediksi
    # Di dataset Iris: 0 adalah Iris Setosa, 1 adalah Iris Versicolor, 2 adalah Iris Virginica
    return result