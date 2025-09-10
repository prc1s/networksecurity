import sys,os
import certifi

from urllib.parse import quote_plus
from dotenv import load_dotenv
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.ml_utils.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_pickle_object

from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

ca = certifi.where()
load_dotenv()
raw_mongodb_username =os.getenv("MONGO_DB_USERNAME")
raw_mongodb_password = os.getenv("MONGO_DB_PASSWORD")

mongodb_username = quote_plus(raw_mongodb_username)
mongodb_password = quote_plus(raw_mongodb_password)

MONGO_DB_URL=f"mongodb+srv://{mongodb_username}:{mongodb_password}@cluster0.yfmjcsk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training is Successful!")
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        final_model = load_pickle_object("final_models/model.pkl")
        preprocessor = load_pickle_object("final_models/preprocessor.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request":request, "table":table_html})
    except Exception as e:
        logger.exception(NetworkSecurityException(e,sys))
        raise NetworkSecurityException(e,sys)
if __name__ == "__main__":
    app_run(app,host="0.0.0.0",port=4141)
