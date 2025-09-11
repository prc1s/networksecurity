# PhishGuard AI

A machine learning system I built to detect phishing websites. This project started as a way to learn MLOps practices and ended up being a complete pipeline that can actually identify malicious websites in real-time.

## What it does

The system analyzes website features like URL structure, SSL certificates, domain age, and other security indicators to determine if a site is legitimate or a phishing attempt. I trained it on 11,055 samples with 31 different features.

## How it works

1. **Data Pipeline**: Pulls data from MongoDB, validates it against a schema, and handles missing values
2. **Model Training**: Tests 5 different algorithms (Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, AdaBoost) and picks the best one
3. **API**: FastAPI web service where you can upload a CSV file and get predictions back
4. **Cloud Storage**: Saves models and artifacts to AWS S3, tracks experiments with MLflow

## Tech stack

- Python 3.10
- scikit-learn for ML models
- FastAPI for the web API
- MongoDB Atlas for data storage
- AWS S3 for model storage
- MLflow for experiment tracking
- Docker for deployment

## Setup

1. Clone the repo and install dependencies:
```bash
git clone <your-repo>
cd NetworkSecurity
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```env
MONGO_DB_USERNAME=your_username
MONGO_DB_PASSWORD=your_password
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

3. Run the training:
```bash
python main.py
```

4. Start the API server:
```bash
python app.py
```

5. Track Models:
    After pressing Train in http://localhost:4141
        '''bash
        mlflow ui
        '''
Then go to http://localhost:4141 and upload a CSV file to get predictions.

## Project structure

```
NetworkSecurity/
├── networksecurity/           # Main package
│   ├── components/           # ML pipeline components
│   ├── pipeline/            # Training and prediction pipelines
│   ├── entity/              # Configuration classes
│   ├── utils/               # Helper functions
│   └── cloud/               # S3 integration
├── data_schema/             # Data validation rules
├── templates/               # HTML for web interface
└── Network_Data/           # Dataset
```

## Key features I implemented

- **Automated ML pipeline**: Data ingestion → validation → transformation → training → deployment
- **Multiple model comparison**: Tests different algorithms and picks the best performer
- **Data validation**: Checks data quality and detects drift between train/test sets
- **Cloud integration**: MongoDB for data, S3 for model storage
- **Web interface**: Upload CSV files and get predictions in a table format
- **Model versioning**: MLflow tracks all experiments and model versions

## What I learned

This project taught me a lot about MLOps - how to structure ML code properly, handle data validation, implement automated pipelines, and deploy models to production. The hardest part was getting the data validation and drift detection working correctly.

## Performance

The system evaluates models based on F1 score, precision, and recall. I used GridSearchCV for hyperparameter tuning and MLflow to track all the experiments. The best performing model gets automatically selected and saved.

## Author

Mohammed Aljowaie  
Email: mbfaj2@gmail.com

---

*Built as a learning project to understand MLOps and cybersecurity applications of machine learning.*