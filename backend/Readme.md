# Project Title

**Final Year Project Backend**

## Description
Provide a brief description of your project, its purpose, and what it aims to achieve.

## Installation
Step-by-step instructions on how to set up the project locally.

```bash
# Navigate to the project directory
cd yourproject

# Install dependencies
npm install
```

## Create a Virtual Environment

### Unix/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

## Install Dependencies
Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Project Strucuture 

``` bash

backend/
├── models/
│   └── xgb_classifier.joblib 
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── schemas.py
├── requirements.txt
└── README.md

``` 

## Usage
Instructions and examples on how to use the project.

```bash
# Start the application
npm start
```

## Model
The project uses an XGBoost classifier trained on the fraud-transcations.txt dataset.

- **Training Script:** Located in `app/models.py`.
- **Model File:** `models/xgb_classifier.joblib`

## API Endpoints

### POST /predict
- **Description:** Predicts whether a transaction is fraudulent.
- **Request Body:** Transaction details adhering to the `schemas.Transaction` schema.
- **Response:** Prediction result.

## Dependencies
All dependencies are listed in `requirements.txt`. Key libraries include:

- **FastAPI:** Web framework for building APIs.
- **Uvicorn:** ASGI server for running FastAPI applications.
- **XGBoost:** Machine learning library for the classifier.
- **Pydantic:** Data validation library used with FastAPI.
