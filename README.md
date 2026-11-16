# Penerapan Convolutional Neural Network untuk Klasifikasi Emosi Wajah Pekerja

## Deskripsi Proyek

Proyek ini merupakan implementasi Convolutional Neural Network (CNN) untuk mengklasifikasikan ekspresi wajah pekerja ke dalam enam kategori, yaitu Angry, Fatigue, Fear, Pain, Sad, dan Surprise.

Penelitian ini bertujuan mengembangkan sistem klasifikasi emosi wajah berbasis pengolahan citra dan deep learning sebagai salah satu pendekatan untuk mengenali ekspresi wajah pekerja.

## Kategori Klasifikasi

Model mengklasifikasikan wajah ke dalam enam kategori:

- Angry (Marah)
- Fatigue (Kelelahan)
- Fear (Takut)
- Pain (Kesakitan)
- Sad (Sedih)
- Surprise (Terkejut)

## Teknologi dan Library

### Model Development
- **Python** – Bahasa pemrograman utama.
- **TensorFlow / Keras** – Pengembangan dan pelatihan model CNN.
- **Pandas & NumPy** – Pengolahan data dan operasi numerik.
- **Matplotlib & Seaborn** – Visualisasi data dan hasil pelatihan.
- **Pillow (PIL)** – Pemrosesan gambar.
- **Plotly** – Visualisasi data interaktif.
- **Scikit-learn** – Evaluasi model menggunakan classification report dan confusion matrix.

### Desktop Application
- **Tkinter** – Pengembangan antarmuka aplikasi desktop.
- **OpenCV** – Pengolahan citra dan akses webcam.
- **TensorFlow / Keras** – Memuat model untuk melakukan prediksi.
- **Pillow (PIL)** – Menampilkan gambar pada antarmuka aplikasi.
- **NumPy** – Pemrosesan array gambar.
- **Pygame** – Pemutaran suara alarm.
- **Threading & Time** – Pengelolaan proses dan waktu pada aplikasi.

## Metode Pengembangan

Model CNN dikembangkan menggunakan citra wajah grayscale berukuran 80 × 80 piksel.

Arsitektur model terdiri dari empat blok convolutional dengan kombinasi Conv2D, ReLU, Batch Normalization, dan Max Pooling. Selanjutnya, model menggunakan Global Average Pooling, Fully Connected Layer dengan 256 neuron dan aktivasi ReLU, Dropout sebesar 0.6, serta Output Layer dengan enam neuron dan aktivasi Softmax untuk menghasilkan prediksi kategori emosi.

Model dilatih menggunakan optimizer Adam dengan learning rate 0.0005.

## Hasil Pengujian

Model memperoleh akurasi sebesar **82,92% pada data pengujian (test set)**.

## Fitur Aplikasi

Aplikasi desktop dikembangkan menggunakan Tkinter dengan fitur:

- Memuat model CNN dalam format `.h5` atau `.keras`.
- Deteksi emosi wajah secara langsung menggunakan webcam.
- Deteksi wajah menggunakan Haar Cascade Classifier dari OpenCV.
- Menampilkan kategori emosi beserta tingkat confidence prediksi.
- Menampilkan probabilitas untuk keenam kategori emosi.
- Peringatan suara menggunakan Pygame ketika kondisi kelelahan terdeteksi berdasarkan ambang prediksi yang ditentukan.
- Kontrol untuk memulai dan menghentikan kamera serta alarm.

## Tujuan Pembelajaran

Melalui proyek ini, saya memperoleh pengalaman dalam pengolahan citra digital, pengembangan model deep learning, evaluasi model, serta implementasi model CNN ke dalam aplikasi desktop.

## Catatan

Sistem ini merupakan proyek akademik dan bukan alat diagnosis psikologis atau medis.
