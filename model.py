import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/train.csv")

y = df["Interest_Rate"]
X = df.drop(["Interest_Rate","Loan_ID"], axis=1)

cat_cols = X.select_dtypes(include="object").columns

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X[cat_cols] = encoder.fit_transform(X[cat_cols])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_val,y_train,y_val = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_val)
mae = mean_absolute_error(y_val,pred)

print("MAE:",mae)

joblib.dump(model,"artifacts/model.pkl")
joblib.dump(scaler,"artifacts/scaler.pkl")
joblib.dump(encoder,"artifacts/encoder.pkl")

print("Model saved")
