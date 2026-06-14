import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  

try:
    dataset = pd.read_csv('iris.data', header=None, sep=',')
except FileNotFoundError:
    print("Error: File 'iris.data' tidak ditemukan di folder ini!")
    print("Mencoba mengambil secara online sebagai cadangan...")
    dataset = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(X_train.shape[1],)),  
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')      
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== Memulai Pelatihan Model ===")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("\n=== Evaluasi Model ===")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

pd.DataFrame(history.history).plot(figsize=(10,6))
plt.title('Grafik Loss dan Accuracy Selama Pelatihan')
plt.xlabel('Epochs')
plt.ylabel('Nilai')
plt.grid(True)
plt.show()

predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1) 

print("\nHasil Prediksi vs Label Asli:")
print("Prediksi  :", predicted_classes)
print("Label Asli:", y_test)

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Spesies Bunga Iris')
plt.show()

def predict_new_data():
    print("\n" + "="*40)
    print("   PREDIKSI DATA BUNGA IRIS BARU   ")
    print("="*40)
    try:
        sepal_length = float(input("Masukkan sepal length (contoh: 5.1): "))
        sepal_width  = float(input("Masukkan sepal width  (contoh: 3.5): "))
        petal_length = float(input("Masukkan petal length (contoh: 1.4): "))
        petal_width  = float(input("Masukkan petal width  (contoh: 0.2): "))
        
        new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        prediction = model.predict(new_data)
        predicted_class = prediction.argmax(axis=1)
        
        predicted_label = label_encoder.inverse_transform(predicted_class)
        print(f"\nHasil Prediksi Kelas: {predicted_label[0]}")
    except ValueError:
        print("Input tidak valid! Pastikan Anda memasukkan angka.")

predict_new_data()