import streamlit as st
import pandas as pd

import joblib
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", page_title="Loan Analytics", page_icon="📊")

# Dark aesthetic
st.markdown("""
<style>
body { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("Loan Interest, Risk and Revenue Analytics Platform")

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
encoder = joblib.load("artifacts/encoder.pkl")

uploaded = st.file_uploader("Upload your csv file")

if uploaded:

    df = pd.read_csv(uploaded)

    ids = df["Loan_ID"]
    df2 = df.drop("Loan_ID",axis=1)

    cat_cols = df2.select_dtypes(include="object").columns
    df2[cat_cols] = encoder.transform(df2[cat_cols])

    X = scaler.transform(df2)

    pred = model.predict(X)

    out = pd.DataFrame()
    out["Loan_ID"] = ids
    out["Predicted_Interest"] = pred
    out["Risk_Score"] = pred * df2["Debt_To_Income"]
    out["Expected_Revenue"] = df2["Loan_Amount_Requested"] * (pred/100)

    # Risk Buckets
    out["Risk_Bucket"] = pd.qcut(out["Risk_Score"], q=3, labels=["Low","Medium","High"])

    ranked = out.sort_values("Expected_Revenue",ascending=False)

    # ---------- KPI ROW ----------

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Avg Interest", round(ranked["Predicted_Interest"].mean(),2))
    c2.metric("Total Revenue", f"{int(ranked['Expected_Revenue'].sum()):,}")
    c3.metric("High Risk Loans", (ranked["Risk_Bucket"]=="High").sum())
    c4.metric("Top 20% Revenue %",
        round(ranked.head(int(len(ranked)*0.2))["Expected_Revenue"].sum() /
        ranked["Expected_Revenue"].sum()*100,2))

    st.divider()

    # ---------- LAYOUT ----------

    left,right = st.columns(2)

    with left:

        fig1 = px.bar(
            ranked.head(20),
            x="Loan_ID",
            y="Expected_Revenue",
            title="Top 20 Loans by Revenue",
            color="Expected_Revenue",
            height=350,
            template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(
            ranked,
            x="Predicted_Interest",
            nbins=25,
            title="Interest Rate Distribution",
            height=350,
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:

        ranked["Bubble_Size"] = ranked["Expected_Revenue"] - ranked["Expected_Revenue"].min() + 1

        fig3 = px.scatter(
            ranked,
            x="Risk_Score",
            y="Expected_Revenue",
            color="Risk_Bucket",
            size="Bubble_Size",
            title="Risk vs Revenue Map",
            height=350,
            template="plotly_dark"
        )

        bucket = ranked.groupby("Risk_Bucket")["Expected_Revenue"].sum().reset_index()

        fig4 = px.pie(
            bucket,
            values="Expected_Revenue",
            names="Risk_Bucket",
            title="Revenue by Risk Bucket",
            template="plotly_dark"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ---------- EXECUTIVE ANALYSIS ----------

    st.subheader("Finance Insights")

    high_value = ranked.head(int(len(ranked)*0.2))
    revenue_share = round(high_value["Expected_Revenue"].sum() /
                          ranked["Expected_Revenue"].sum()*100,2)

    st.write("• Top 20 percent customers generate", revenue_share, "percent of revenue")
    st.write("• High risk loans show higher interest concentration")
    st.write("• Revenue heavily skewed toward small customer segment")
    st.write("• Medium risk group offers best balance between profit and safety")

    st.subheader("Ranked Loan Table")
    st.dataframe(ranked.head(40))

    st.download_button(
        "Download Full Ranked Loans",
        ranked.to_csv(index=False),
        "ranked_loans.csv"
    )
