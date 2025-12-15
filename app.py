import streamlit as st
import pandas as pd
import numpy as np
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

st.set_page_config(
    page_title="Store Insight AI", 
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Style the Metrics/KPIs as Cards */
    [data-testid="stMetric"] {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Center metric labels */
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        font-weight: bold;
        color: #555;
    }

    /* Center metric values */
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
    }

    /* Style the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1E1E1E;
    }
    
    /* Custom container for AI Insight */
    .ai-box {
        background-color: #F0F8FF;
        border-left: 5px solid #0068C9;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


################################################
# 1. Data
################################################
@st.cache_data
def load_data():
    df = pd.read_excel("./data.xlsx")
    return df

# ################################################
# 2. FEATURE ENGINEERING 
################################################
def process_retail_metrics(df):
    # 1. The Retail Equation
    df['conversion_rate'] = df['gross_transactions'] / df['traffic']
    df['upt'] = df['gross_quantity'] / df['gross_transactions'] 
    df['aup'] = df['net_sales'] / df['gross_quantity'] 

    # 2. Network Aggregation (Rest of Marker)
    network_df = df.groupby(['Year', 'Week'])[['net_sales', 'traffic', 'gross_transactions']].sum().reset_index()
    
    return df, network_df

def get_store_snapshot(df, network_df, store_name, week, current_year=2025):
    """
    Prepares the specific context for the LLM. 
    Crucially, this handles the YoY join and Anomaly flagging.
    """
    prev_year = current_year - 1
    
    #  Current Year
    curr_data = df[(df['Store Name'] == store_name) & (df['Year'] == current_year) & (df['Week'] == week)].iloc[0]
    
    # Previous Year (Baseline)
    try:
        prev_data = df[(df['Store Name'] == store_name) & (df['Year'] == prev_year) & (df['Week'] == week)].iloc[0]
        has_baseline = True
    except IndexError:
        has_baseline = False
        prev_data = None

    net_curr = network_df[(network_df['Year'] == current_year) & (network_df['Week'] == week)].iloc[0]
    net_prev = network_df[(network_df['Year'] == prev_year) & (network_df['Week'] == week)].iloc[0]
    

    net_avg_sales_curr = (net_curr['net_sales'] - curr_data['net_sales']) / 9
    net_avg_sales_prev = (net_prev['net_sales'] - prev_data['net_sales']) / 9

    
    # 1. YoY Deltas
    if has_baseline and prev_data['net_sales'] > 0:
        sales_yoy = (curr_data['net_sales'] - prev_data['net_sales']) / prev_data['net_sales']
        traffic_yoy = (curr_data['traffic'] - prev_data['traffic']) / prev_data['traffic']
        conv_yoy = (curr_data['conversion_rate'] - prev_data['conversion_rate']) / prev_data['conversion_rate']
        network_yoy = (net_avg_sales_curr - net_avg_sales_prev) / net_avg_sales_prev
    else:
        sales_yoy, traffic_yoy, conv_yoy, network_yoy = 0, 0, 0, 0

    # 2. Gap to Network
    vs_network_gap = sales_yoy - network_yoy

    # 3. Anomaly Detection (Privacy-safe signals)
    baseline_abnormal = False
    if has_baseline and prev_data['traffic'] < 200:
        baseline_abnormal = True

    # Structured Context Dictionary

    context = {
        "store": store_name,
        "week": week,
        "metrics": {
            "sales_current": round(curr_data['net_sales'], 2),
            "traffic_current": int(curr_data['traffic']),            
            "conversion_current": round(curr_data['conversion_rate'] * 100, 2), 
            "sales_yoy_pct": round(sales_yoy * 100, 1),
            "traffic_yoy_pct": round(traffic_yoy * 100, 1),
            "conversion_yoy_pct": round(conv_yoy * 100, 1),
        },
        "network_context": {
            "network_sales_yoy_pct": round(network_yoy * 100, 1),
            "gap_to_network_pts": round(vs_network_gap * 100, 1)
        },
        "flags": {
            "baseline_abnormal": baseline_abnormal,
            "is_positive_trend": sales_yoy > 0
        }
    }

    return context

################################################
# 3. LLM INTEGRATION
################################################

def generate_llm_insight(context, api_key):
    """
    Generates the analysis.
    (to ensure the prototype works for everyone reviewing).
    """
    
    system_prompt = """
    You are a Retail Performance Analyst. 
    Analyze the JSON data. Output a 3-5 line summary for the Store Manager.
    
    RULES:
    1. CHECK BASELINE: If 'baseline_abnormal' is True, start by warning that YoY comparison is invalid due to store closure/issues last year.
    2. ROOT CAUSE: Use the Retail Equation. If Sales are down, is it Traffic (Marketing) or Conversion (Operations)?
    3. CONTEXT: Compare 'sales_yoy_pct' vs 'network_sales_yoy_pct'. If store is down but network is down more, mention resilience.
    4. TONE: Professional, concise, action-oriented.
    """
    
    user_prompt = str(context)

    client = genai.Client(api_key= api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt),
        contents= user_prompt
    )

    return response.text




################################################
# 4. USER INTERFACE (Streamlit)
################################################

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Load Data
raw_df = load_data()
df, network_df = process_retail_metrics(raw_df)

st.title("🛍️ Store Performance")

with st.container(border=True):
    col_sel_1, col_sel_2 = st.columns(2)
    
    with col_sel_1:
        selected_store = st.selectbox(
            "Select Store Location", 
            df['Store Name'].unique(),
            index=0
        )
    
    with col_sel_2:
        selected_week = st.slider(
            "Fiscal Week (2025)", 
            min_value=1, 
            max_value=48, 
            value=42
        )

context = get_store_snapshot(df, network_df, selected_store, selected_week)

st.markdown("### Performance Snapshot")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

def format_delta(val):
    return f"{val}%"

with kpi1:
    st.metric(
        "Net Sales", 
        f"${context['metrics']['sales_current']:,.0f}", 
        format_delta(context['metrics']['sales_yoy_pct'])
    )
with kpi2:
    st.metric(
        "Traffic", 
        f"{context['metrics']['traffic_current']:,}", 
        format_delta(context['metrics']['traffic_yoy_pct'])
    )
with kpi3:
    st.metric(
        "Conversion", 
        f"{context['metrics']['conversion_current']}%", 
        format_delta(context['metrics']['conversion_yoy_pct'])
    )
with kpi4:
    st.metric(
        "Vs Network", 
        f"{context['network_context']['gap_to_network_pts']} pts", 
        delta_color="off" if context['network_context']['gap_to_network_pts'] == 0 else "normal"
    )

st.divider()

st.subheader("🤖 AI Analyst & Signals")

col_ai, col_signals = st.columns([2, 1])

with col_ai:
    st.caption("Automated Insight Generation")
    insight_container = st.container(border=True)
    with insight_container:
        if api_key:
            insight_text = generate_llm_insight(context, api_key=api_key)
            if insight_text:
                st.markdown(insight_text)
            else:
                st.info("Analysis generated ")
        else:
            st.error("Gemini API Key not found.")

with col_signals:
    st.caption("Data Validity Checks")
    st.info("""
    **Signal Detection:**
    
    **Baseline Check:**  
    {}
    
    **Network Alignment:**  
    {}
    """.format(
        "⚠️ Abnormal (2024 Data Issue)" if context['flags']['baseline_abnormal'] else "✅ Valid Comparison",
        "Above Market Average" if context['network_context']['gap_to_network_pts'] > 0 else "Below Market Average"
    ))