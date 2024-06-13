
import os
from dotenv import load_dotenv
from io import BytesIO
from fastapi import FastAPI, HTTPException, Depends, status, Path
from pydantic import BaseModel
from typing import Annotated, Dict
from sklearn.calibration import LabelEncoder
from sqlalchemy import text
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from google.cloud import storage
from tensorflow import keras
from keras.models import load_model
import tempfile
import numpy as np
import pandas as pd


 # Load the environment variables from the .env file
load_dotenv() 

# Define a function to load the model from GCS
def load_model_from_gcs(bucket_name, model_path):

    """Loads a Keras model from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
   
    blob = bucket.blob(model_path)

    # This is the key part: Access the data directly from GCS
    model_data = blob.download_as_bytes()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
        temp_file.write(model_data)
        temp_file_path = temp_file.name

    # Load the model from bytes using load_model
    model = load_model(temp_file_path)

    # Delete the temporary file
    os.remove(temp_file_path)
    return model




app = FastAPI()
bucket = "lidm_211"
tiu_model = "model_tiu3"
tkp_model = "model_tkp3"
twk_model = "model_twk3"
model_tiu = "model/"+tiu_model+".h5"
model_tkp = "model/"+tkp_model+".h5"
model_twk = "model/"+twk_model+".h5"
modelTIU = load_model_from_gcs(bucket_name=bucket,model_path=model_tiu )
modelTWK = load_model_from_gcs(bucket_name=bucket,model_path=model_twk )
modelTKP = load_model_from_gcs(bucket_name=bucket,model_path=model_tkp )
data_saran = pd.read_csv("https://storage.googleapis.com/lidm_211/model/data_saran.csv")
mock = pd.read_csv("https://storage.googleapis.com/lidm_211/tryout/mock.csv")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Skor minimal lulus
min_tiu_score = 80
min_twk_score = 65
min_tkp_score = 166


label_encoder_tiu = LabelEncoder()
label_encoder_twk = LabelEncoder()
label_encoder_tkp = LabelEncoder()

y_tiu =mock ['label_tiu']

y_twk =mock ['label_twk']

y_tkp =mock ['label_tkp']

y_tiu=label_encoder_tiu.fit_transform(y_tiu)
y_twk=label_encoder_twk.fit_transform(y_twk)
y_tkp=label_encoder_tkp.fit_transform(y_tkp)
print(label_encoder_tiu)

# Fungsi untuk memberikan feedback yang menggabungkan materi yang harus ditingkatkan dan detail skor
def predict_and_feedback(user_data):
    # # Predict for TIU
    tiu_predictions = modelTIU.predict(user_data[['kesalahan_analogi_verbal', 'kesalahan_silogisme', 'kesalahan_analitis', 'kesalahan_berhitung', 'kesalahan_deret_angka', 'kesalahan_perbandingan_kuantitatif', 'kesalahan_soal_cerita', 'kesalahan_ketidaksamaan', 'kesalahan_serial', 'kesalahan_analogi_figural']])
    tiu_predictions_labels = label_encoder_tiu.inverse_transform(np.argmax(tiu_predictions, axis=1))

    # Predict for TWK
    twk_predictions = modelTWK.predict(user_data[['kesalahan_nasionalisme', 'kesalahan_integritas', 'kesalahan_bela_negara', 'kesalahan_pilar_negara', 'kesalahan_bahasa_indonesia']])
    twk_predictions_labels = label_encoder_twk.inverse_transform(np.argmax(twk_predictions, axis=1))

    # Predict for TKP
    tkp_predictions = modelTKP.predict(user_data[['skor_pelayanan_publik', 'skor_jejaring_kerja', 'skor_sosial_budaya', 'skor_tik', 'skor_profesionalisme', 'skor_anti_radikalisme']])
    tkp_predictions_labels = label_encoder_tkp.inverse_transform(np.argmax(tkp_predictions, axis=1))

    user_data['total_tiu_score'] = 175 - (user_data[['kesalahan_analogi_verbal', 'kesalahan_silogisme', 'kesalahan_analitis', 'kesalahan_berhitung', 'kesalahan_deret_angka', 'kesalahan_perbandingan_kuantitatif', 'kesalahan_soal_cerita', 'kesalahan_ketidaksamaan', 'kesalahan_serial', 'kesalahan_analogi_figural']].sum(axis=1) * 5)
    user_data['total_twk_score'] = 150 - (user_data[['kesalahan_nasionalisme', 'kesalahan_integritas', 'kesalahan_bela_negara', 'kesalahan_pilar_negara', 'kesalahan_bahasa_indonesia']].sum(axis=1) * 5)
    user_data['total_tkp_score'] = user_data[['skor_pelayanan_publik', 'skor_jejaring_kerja', 'skor_sosial_budaya', 'skor_tik', 'skor_profesionalisme', 'skor_anti_radikalisme']].sum(axis=1)

    tiu_errors = user_data[['kesalahan_analogi_verbal', 'kesalahan_silogisme', 'kesalahan_analitis', 'kesalahan_berhitung', 'kesalahan_deret_angka', 'kesalahan_perbandingan_kuantitatif', 'kesalahan_soal_cerita', 'kesalahan_ketidaksamaan', 'kesalahan_serial', 'kesalahan_analogi_figural']]
    twk_errors = user_data[['kesalahan_nasionalisme', 'kesalahan_integritas', 'kesalahan_bela_negara', 'kesalahan_pilar_negara', 'kesalahan_bahasa_indonesia']]
    tkp_scores = user_data[['skor_pelayanan_publik', 'skor_jejaring_kerja', 'skor_sosial_budaya', 'skor_tik', 'skor_profesionalisme', 'skor_anti_radikalisme']]

    tiu_feedback_col = tiu_errors.columns[np.argmax(tiu_errors.values)]
    twk_feedback_col = twk_errors.columns[np.argmax(twk_errors.values)]
    tkp_feedback_col = tkp_scores.columns[np.argmin(tkp_scores.values)]
    tkp_best_col = tkp_scores.columns[np.argmax(tkp_scores.values)]

    tiu_feedback = data_saran[tiu_feedback_col].values[0]
    twk_feedback = data_saran[twk_feedback_col].values[0]
    tkp_feedback = data_saran[tkp_feedback_col].values[0]

    tiu_feedback_col_clean = tiu_feedback_col.replace("kesalahan_", "").replace("_", " ").title()
    twk_feedback_col_clean = twk_feedback_col.replace("kesalahan_", "").replace("_", " ").title()
    tkp_feedback_col_clean = tkp_feedback_col.replace("skor_", "").replace("_", " ").title()
    tkp_best_col_clean = tkp_best_col.replace("skor_", "").replace("_", " ").title()
    return {
        "tiu":(f"Saat ini skor untuk TIU Anda adalah {tiu_predictions_labels[0]}  dengan nilai {user_data['total_tiu_score'].values[0]} dari 175\n"
               f"Di bagian TIU Anda harus meningkatkan materi {tiu_feedback_col_clean}\nBerikut ini saran untuk meningkatkan nilai Anda: \n{tiu_feedback}\n\n"
        ),
        "twk":(
             f"Saat ini nilai TWK Anda adalah {twk_predictions_labels[0]} dengan nilai {user_data['total_twk_score'].values[0]} dari 150\n"
            f"Di bagian TWK Anda harus meningkatkan materi {twk_feedback_col_clean}\nBerikut ini saran untuk meningkatkan nilai Anda: \n{twk_feedback}\n\n"
        ),
        "tkp":(
            f"Saat ini nilai TKP Anda adalah {tkp_predictions_labels[0]} dengan nilai {user_data['total_tkp_score'].values[0]} dari 225.\n"
            f"Di bagian TKP Anda harus meningkatkan materi {tkp_feedback_col_clean}\nBerikut ini saran untuk meningkatkan nilai Anda: \n{tkp_feedback}"
        )
    }



db_depedency = Annotated[Session,Depends(get_db)]

class SKDInput(BaseModel):
    kesalahan_analogi_verbal: int
    kesalahan_silogisme: int
    kesalahan_analitis: int
    kesalahan_berhitung: int
    kesalahan_deret_angka: int
    kesalahan_perbandingan_kuantitatif: int
    kesalahan_soal_cerita: int
    kesalahan_ketidaksamaan: int
    kesalahan_serial: int
    kesalahan_analogi_figural: int
    kesalahan_nasionalisme: int
    kesalahan_integritas: int
    kesalahan_bela_negara: int
    kesalahan_pilar_negara: int
    kesalahan_bahasa_indonesia: int
    skor_pelayanan_publik: int
    skor_jejaring_kerja: int
    skor_sosial_budaya: int
    skor_tik: int
    skor_profesionalisme: int
    skor_anti_radikalisme: int







@app.get("/skd_analysis/{tryout_id}/{account_id}",status_code=status.HTTP_200_OK)
async def tryout_analysis(tryout_id:str = Path(..., title="The id of the tryout"), account_id:str = Path(...,title="The id of the account"),db: SessionLocal = Depends(get_db)):
    try:
        query=text("SELECT sb.*,(SELECT skd.`subCategory_score` FROM `skd_analysis` skd WHERE skd.`account_id`=:account_id AND skd.`tryout_id`=:tryout_id) as score FROM `subcategory` sb;")
        results = db.execute(query,{"account_id":account_id,"tryout_id":tryout_id})
     
        data = results.mappings().all()
        print(len(data))
        datanew:SKDInput ={
            "kesalahan_analogi_verbal":[data[0]['score'] or 0] ,
            "kesalahan_silogisme":[data[1]['score'] or 0] ,
            "kesalahan_analitis":[data[2]['score'] or 0] ,
            "kesalahan_berhitung":[data[3]['score']or 0] ,
            "kesalahan_deret_angka":[data[4]['score']or 0] ,
            "kesalahan_perbandingan_kuantitatif":[data[5]['score'] or 0] ,
            "kesalahan_soal_cerita":[data[6]['score'] or 0] ,
            "kesalahan_ketidaksamaan":[data[7]['score'] or 0] ,
            "kesalahan_serial":[data[8]['score'] or 0] ,
            "kesalahan_analogi_figural":[(data[7]['score'] or 0 ) + (data[8]['score'] or 0 )],
            "kesalahan_nasionalisme":[data[9]['score']or 0] ,
            "kesalahan_integritas":[data[10]['score'] or 0] ,
            "kesalahan_bela_negara":[data[11]['score'] or 0] ,
            "kesalahan_pilar_negara":[data[12]['score'] or 0] ,
            "kesalahan_bahasa_indonesia":[data[13]['score'] or 0] ,
            "skor_pelayanan_publik":[data[14]['score'] or 0],
            "skor_jejaring_kerja":[data[15]['score'] or 0] ,
            "skor_sosial_budaya":[data[16]['score'] or 0] ,
            "skor_tik":[data[17]['score'] or 0] ,
            "skor_profesionalisme":[data[18]['score'] or 0]  ,
            "skor_anti_radikalisme":[data[19]['score'] or 0] 
        }
       
     

       

        # tiuPredict = modelTIU.predict(input_data)
        # twkPredict = modelTWK.predict(input_data)
        # tkpPredict = modelTKP.predict(input_data) 
        # print(tiuPredict)
        print(datanew)
        user_data = pd.DataFrame(datanew)

        res =predict_and_feedback(user_data=user_data)
        
       
        return {"feedback":res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))