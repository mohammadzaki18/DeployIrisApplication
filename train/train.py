from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Memuat dataset Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

print("model running")

# Membagi dataset menjadi data latihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi fitur
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Membuat objek classifier SVM
clf = SVC(kernel='linear')

# Melatih model dengan data latihan
clf.fit(X_train, y_train)

# Memprediksi hasil dengan data uji
y_pred = clf.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Menyimpan model
filename = './model/iris_svm_model.sav'
joblib.dump(clf, filename)

# Menyimpan scaler
scaler_filename = 'scaler.save'
joblib.dump(sc, scaler_filename)