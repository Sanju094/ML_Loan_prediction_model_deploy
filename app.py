from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
import joblib

# Load the trained CatBoost model
model_path = "catboost_model.cbm"  # Replace with the actual path to your saved model
cb_model = CatBoostClassifier()
cb_model.load_model(model_path)

# Define FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model using Pydantic
class LoanPredictionInput(BaseModel):
     no_of_dependents: int
     education: int
     self_employed: int
     income_annum: float
     loan_amount: float
     loan_term: float
     cibil_score: float
     residential_assets_value: float
     commercial_assets_value: float
     luxury_assets_value: float
     bank_asset_value: float

# Update predict_loan_status function to use the correct feature names
@app.post("/predict")
async def predict_loan_status(data: LoanPredictionInput):
    try:
        input_data = pd.DataFrame({
            ' no_of_dependents': [data.no_of_dependents],
            ' education': [data.education],
            ' self_employed': [data.self_employed],
            ' income_annum': [data.income_annum],
            ' loan_amount': [data.loan_amount],
            ' loan_term': [data.loan_term],
            ' cibil_score': [data.cibil_score],
            ' residential_assets_value': [data.residential_assets_value],
            ' commercial_assets_value': [data.commercial_assets_value],
            ' luxury_assets_value': [data.luxury_assets_value],
            ' bank_asset_value': [data.bank_asset_value],
        })

        prediction = int(cb_model.predict(input_data)[0])
        return {"Loan_Status_Prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
