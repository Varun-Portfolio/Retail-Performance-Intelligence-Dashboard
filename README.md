# Retail-Performance-Intelligence-Dashboard
### Built by Varun Mehta — Data Analyst

A full-stack business intelligence dashboard built with Python and Streamlit.

## 📊 Features
- **7 analytical tabs** — Overview, Category, Store, Channel, Profitability, Forecast, Insights
- **Revenue & margin tracking** across stores, categories and channels
- **6-month revenue forecast** using Linear Regression (sklearn)
- **Auto-generated insights** with strategic recommendations
- **Strategic time periods** — YTD, QTD, Last 12 Months, Full Range

## 🛠 Tech Stack
Python · Pandas · NumPy · Plotly · Streamlit · scikit-learn

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run App/app.py
## 📁 Data
Place your `sales_data.csv` in the `data/` folder.
Columns required: Date, Store_Id, Channel, Category, Product_Name, Units_Sold, Revenue, Cost, Return_Units
