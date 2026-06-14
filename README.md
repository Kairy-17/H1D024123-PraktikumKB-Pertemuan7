# Pertemuan 7: Jaringan Syaraf Tiruan (JST) 2 - Klasifikasi Spesies Bunga Iris

## 📝 Deskripsi Tugas & Studi Kasus
Praktikum ini berfokus pada implementasi konsep Jaringan Syaraf Tiruan (JST) jenis *Multi-Layer Perceptron* (MLP) untuk melakukan klasifikasi tiga spesies bunga Iris (Iris Setosa, Iris Versicolor, dan Iris Virginica). Model dilatih untuk mengenali pola spesies berdasarkan empat fitur morfologi utama, yaitu panjang sepal, lebar sepal, panjang petal, dan lebar petal dari dataset lokal `iris.data`.

## 🧠 Arsitektur Model JST
Model dibangun secara **Sequential** menggunakan library **TensorFlow** dan **Keras** dengan detail susunan layer sebagai berikut:
* **Input Layer**: Menerima 4 fitur morfologi bunga Iris.
* **Hidden Layer 1**: Dense Layer (1000 neuron, fungsi aktivasi `ReLU`).
* **Hidden Layer 2**: Dense Layer (500 neuron, fungsi aktivasi `ReLU`).
* **Hidden Layer 3**: Dense Layer (300 neuron, fungsi aktivasi `ReLU`).
* **Output Layer**: Dense Layer (3 neuron, fungsi aktivasi `Softmax`) untuk mengklasifikasikan ke dalam 3 kelas spesies target.

## 📊 Alur Kerja & Fitur Program
1. **Load Dataset**: Membaca dataset Iris secara lokal dari file teks `iris.data`.
2. **Data Preprocessing**: Mengonversi label dengan `LabelEncoder` dan membagi data latih/uji dengan rasio 80:20.
3. **Kompilasi Model**: Menggunakan optimizer `Adam` dan loss `sparse_categorical_crossentropy`.
4. **Pelatihan & Evaluasi**: Model dilatih 50 epoch, menampilkan grafik metrik `matplotlib`, dan *Confusion Matrix* `seaborn`.
5. **Prediksi Interaktif**: Input data sepal/petal baru secara manual di terminal untuk mengecek spesies bunga.

## 🚀 Cara Menjalankan Program

Salin dan ikuti seluruh baris perintah di dalam satu kotak di bawah ini secara berurutan:

```bash
# LANGKAH 1: Install semua library pendukung yang diperlukan
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# LANGKAH 2: Pastikan file dataset bernama 'iris.data' sudah 
# diletakkan di folder yang sama dengan file skrip 'jst_iris.py'

# LANGKAH 3: Jalankan program utama Anda
python jst_iris.py
