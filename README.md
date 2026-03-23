# Email Spam Classification API

API Machine Learning berbasis **FastAPI** untuk mendeteksi apakah sebuah teks atau email merupakan **Spam** atau **Bukan Spam (Ham)**. Proyek ini menggunakan algoritma *Naive Bayes* dan *TF-IDF Vectorizer* dari Scikit-Learn.

## 🌟 Fitur Utama
* **Fast Inference:** Prediksi teks secara instan menggunakan FastAPI.
* **Custom Threshold:** Sensitivitas model diatur pada batas `45%` (0.45) untuk mendeteksi *spam* dengan lebih agresif, meminimalisir lolosnya pesan penipuan.
* **Confidence Score:** Mengembalikan probabilitas persentase (seberapa yakin model) untuk masing-masing kelas (Spam vs Ham).

## 📂 Struktur Repositori
```text
email-spam-clasification/
├── dataset/
│   └── emails.csv                 # Dataset mentah yang digunakan untuk melatih model
├── email_spam_api/
│   ├── main.py                    # Script utama FastAPI
│   └── model/
│       ├── model_nb_final.joblib  # Model klasifikasi Naive Bayes yang sudah dilatih
│       └── tfidf_vectorizer.joblib# Model pemrosesan teks TF-IDF
├── training_model/
│   └── requirements.txt           # Daftar dependensi dan library Python
└── README.md
```

## 🚀 Cara Menjalankan Proyek (Local Development)
### 1. Clone Repositori
```text
git clone [https://github.com/hikalputra12/email-spam-clasification.git](https://github.com/hikalputra12/email-spam-clasification.git)
cd email-spam-clasification
```
### 2. Install Dependensi
Sangat disarankan untuk menggunakan virtual environment sebelum menginstal library pendukung.

``` 
pip install -r training_model/requirements.txt
```
### 3. Jalankan Server FastAPI
Masuk ke direktori API dan jalankan server menggunakan uvicorn.

```
cd email_spam_api
uvicorn main:app --host localhost --port 8000 --reload
API sekarang akan berjalan di http://localhost:8000.
```

## 📖 Dokumentasi API
Endpoint: POST /predict
Digunakan untuk memprediksi kategori dari teks input.

Request Header:

Content-Type: application/json

Request Body:
```
JSON
{
  "text": "Masukkan teks email atau pesan di sini"
}
```
Contoh Response Sukses (200 OK):

```
JSON
{
  "status": "success",
  "input_text": "Masukkan teks email atau pesan di sini",
  "prediction_code": 1,
  "prediction_label": "Spam",
  "applied_threshold": "45.0%",
  "confidence": {
    "kemungkinan_ham": "20.15%",
    "kemungkinan_spam": "79.85%"
  }
}
```
Keterangan Response:
* prediction_code: 0 berarti Bukan Spam (Ham), 1 berarti Spam.

* prediction_label: Interpretasi dari kode prediksi.

* applied_threshold: Batas probabilitas minimal untuk dianggap sebagai spam.

* confidence: Nilai keyakinan model terhadap prediksinya.

## 🛠️ Teknologi yang Digunakan
* Python 3

* FastAPI & Uvicorn (Backend API)

* Scikit-Learn & Joblib (Machine Learning Pipeline)

* Pandas & Imbalanced-learn (Pengolahan Data & Oversampling)

