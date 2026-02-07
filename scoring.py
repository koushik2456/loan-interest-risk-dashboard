import pandas as pd
import joblib

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
encoder = joblib.load("artifacts/encoder.pkl")

df = pd.read_csv("data/test.csv")

loan_ids = df["Loan_ID"]

df = df.drop("Loan_ID",axis=1)

cat_cols = df.select_dtypes(include="object").columns

df[cat_cols] = encoder.transform(df[cat_cols])

X = scaler.transform(df)

pred_rate = model.predict(X)

result = pd.DataFrame()
result["Loan_ID"] = loan_ids
result["Predicted_Interest"] = pred_rate
result["Risk_Score"] = pred_rate * df["Debt_To_Income"]
result["Expected_Revenue"] = df["Loan_Amount_Requested"] * (pred_rate/100)

ranked = result.sort_values("Expected_Revenue",ascending=False)

ranked.to_csv("ranked_customers.csv",index=False)

print("ranked_customers.csv created")
