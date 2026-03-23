from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# 1. Definisikan Skema Input Data
class TextInput(BaseModel):
    text: str

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Spam Detection API with Custom Threshold", 
    description="API Inference untuk memprediksi apakah teks adalah Spam atau Ham dengan sensitivitas yang bisa diatur.",
    version="1.2"
)

# 2. Muat Artifacts (Model dan Vectorizer)
try:
    model = joblib.load('model/model_nb_final.joblib') 
    tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.joblib') 
    print("✅ Model dan Vectorizer berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat file joblib: {e}")

# Pengaturan Sensitivitas Model (Threshold)
# Jika probabilitas spam > 45%, maka langsung dianggap Spam
SPAM_THRESHOLD = 0.45 

# Endpoint Prediksi
@app.post("/predict")
def predict_text_category(data: TextInput):
    try:
        input_text = data.text
        
        # 3. Transformasi teks menjadi matriks angka
        text_vectorized = tfidf_vectorizer.transform([input_text])
        
        # 4. Hitung Probabilitas (Seberapa yakin modelnya?)
        probabilitas = model.predict_proba(text_vectorized)
        
        # Ekstrak nilai desimal probabilitas
        prob_desimal_ham = probabilitas[0][0]
        prob_desimal_spam = probabilitas[0][1]
        
        # Ubah ke format persentase untuk respons API
        prob_ham_persen = round(prob_desimal_ham * 100, 2)
        prob_spam_persen = round(prob_desimal_spam * 100, 2)
        
        # 5. Logika Prediksi Kustom berdasarkan Threshold
        if prob_desimal_spam >= SPAM_THRESHOLD:
            hasil_prediksi = 1
            status = "Spam"
        else:
            hasil_prediksi = 0
            status = "Bukan Spam (Ham)"
        
        # Respon API
        return {
            "status": "success",
            "input_text": input_text,
            "prediction_code": hasil_prediksi,
            "prediction_label": status,
            "applied_threshold": f"{SPAM_THRESHOLD * 100}%",
            "confidence": {
                "kemungkinan_ham": f"{prob_ham_persen}%",
                "kemungkinan_spam": f"{prob_spam_persen}%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)