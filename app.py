import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# PERFORMANCE BOOST (CACHE)
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/indian_store.csv", encoding='latin1')

@st.cache_resource
def load_model():
    return joblib.load("models/sales_model.pkl")

df = load_data()
model = load_model()

# -----------------------------
# DATE HANDLING
# -----------------------------

date_col = None
for col in ['Order Date', 'Date', 'Order_Date']:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    st.error("No Date column found in dataset")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])

df['Year'] = df[date_col].dt.year
df['Month'] = df[date_col].dt.month

df.dropna(inplace=True)

# -----------------------------
# SALES COLUMN DETECTION
# -----------------------------

sales_col = None
for col in ['Sales', 'Revenue', 'Amount']:
    if col in df.columns:
        sales_col = col
        break

if sales_col is None:
    st.error("No Sales column found")
    st.stop()

# -----------------------------
# SIDEBAR INPUT
# -----------------------------

st.sidebar.title("⚙ Prediction Settings")

year = st.sidebar.selectbox("Year", sorted(df['Year'].unique()))
month = st.sidebar.slider("Month", 1, 12, 6)
quantity = st.sidebar.slider("Quantity", 1, 20, 5)
discount = st.sidebar.slider("Discount", 0.0, 1.0, 0.1)

# -----------------------------
# PREDICTION
# -----------------------------

input_df = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Quantity': [quantity],
    'Discount': [discount]
})

prediction = model.predict(input_df)[0]

# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #00FFAA;'>
     AI Indian Store Sales Forecasting Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align: center; color: #FFFFFF;'>
        Fast & Optimized Machine Learning Dashboard
    </h3>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# KPI METRICS
# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Predicted Sales", f"₹ {prediction:,.2f}")
col2.metric("Total Sales", f"₹ {df[sales_col].sum():,.0f}")
col3.metric("Average Sales", f"₹ {df[sales_col].mean():,.0f}")

# -----------------------------
# MONTHLY SALES
# -----------------------------

@st.cache_data
def monthly_data(df, sales_col):
    return df.groupby('Month')[sales_col].sum().reset_index()

monthly_sales = monthly_data(df, sales_col)

fig1 = px.line(monthly_sales, x='Month', y=sales_col, markers=True,
               title="Monthly Sales Trend")

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# CATEGORY ANALYSIS (SAFE)
# -----------------------------

category_col = None
for col in ['Category', 'Product Category', 'Sub-Category']:
    if col in df.columns:
        category_col = col
        break

if category_col:
    cat_data = df.groupby(category_col)[sales_col].sum().reset_index()

    fig2 = px.bar(cat_data, x=category_col, y=sales_col,
                  title="Category-wise Sales")

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# REGION ANALYSIS (SAFE)
# -----------------------------

region_col = None
for col in ['Region', 'State', 'City']:
    if col in df.columns:
        region_col = col
        break

if region_col:
    region_data = df.groupby(region_col)[sales_col].sum().reset_index()

    fig3 = px.pie(region_data, names=region_col, values=sales_col,
                  title="Region-wise Sales")

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# FUTURE FORECAST
# -----------------------------

future_months = list(range(1, 13))

future_df = pd.DataFrame({
    'Year': [year] * 12,
    'Month': future_months,
    'Quantity': [quantity] * 12,
    'Discount': [discount] * 12
})

future_pred = model.predict(future_df)

forecast_df = pd.DataFrame({
    'Month': future_months,
    'Predicted Sales': future_pred
})

fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=forecast_df['Month'],
    y=forecast_df['Predicted Sales'],
    mode='lines+markers',
    name='Forecast'
))

fig4.update_layout(title="Future Sales Forecast")

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# INSIGHTS
# -----------------------------

st.subheader("🧠 AI Insights")

if prediction > df[sales_col].mean():
    st.success("Above Average Sales Expected 🚀")
else:
    st.warning("Below Average Sales Expected ⚠")

# -----------------------------
# DOWNLOAD
# -----------------------------

st.download_button(
    "📥 Download Forecast",
    forecast_df.to_csv(index=False),
    "forecast.csv",
    "text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")
st.caption("Made with Bharti Sahu")