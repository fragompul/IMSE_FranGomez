# streamlit_app_v2.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# load dataset:
df = pd.read_csv("sigma_delta_results.csv")

# remove infinite values from SNR:
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["SNR"])

# Streamlit page configuration:
st.set_page_config(page_title="SDM analysis", layout="wide")

# title:
st.title("Sigma-Delta Modulator analysis")

# sidebar filters:
st.sidebar.header("Filter options")

# select modulator order:
if "Order" in df.columns:
    orders = st.sidebar.multiselect("Select orders:", sorted(df["Order"].unique()), default=df["Order"].unique())

# select OSR range:
osr_min, osr_max = float(df["OSR"].min()), float(df["OSR"].max())
osr_range = st.sidebar.slider("Select OSR range:", osr_min, osr_max, (osr_min, osr_max))

# select quantization levels:
if "Levels" in df.columns:
    levels = st.sidebar.multiselect("Select quantization levels:", sorted(df["Levels"].unique()), default=df["Levels"].unique())

# select Hinf gain:
if "Hinf" in df.columns:
    hinf_range = st.sidebar.slider("Select Hinf gain range:", float(df["Hinf"].min()), float(df["Hinf"].max()), 
                                   (float(df["Hinf"].min()), float(df["Hinf"].max())))

# select architecture:
if "Form" in df.columns:
    architectures = st.sidebar.multiselect("Select architectures:", sorted(df["Form"].unique()), default=df["Form"].unique())

# select minimum SNR:
if "SNR" in df.columns:
    snr_min, snr_max = float(df["SNR"].min()), float(df["SNR"].max())
    snr_threshold = st.sidebar.slider("Select minimum SNR:", snr_min, snr_max, snr_min)

# apply filters:
df_filtered = df[df["Order"].isin(orders) & df["OSR"].between(*osr_range)]

if "Form" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Form"].isin(architectures)]

if "Hinf" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Hinf"].between(*hinf_range)]

if "Levels" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Levels"].isin(levels)]

if "SNR" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["SNR"] >= snr_threshold]

# display filtered data:
st.write(f"Filtered data: {len(df_filtered)} entries")
st.dataframe(df_filtered)

# charts:
col1, col2 = st.columns(2)

# SNR vs OSR:
with col1:
    st.subheader("SNR vs OSR")
    fig = px.scatter(df_filtered, x="OSR", y="SNR", color="Levels" if "Levels" in df_filtered.columns else None,
                     log_x=True, title="SNR vs OSR", labels={"OSR": "OSR", "SNR": "SNR (dB)"})
    st.plotly_chart(fig, use_container_width=True)

# correlation heatmap:
with col2:
    st.subheader("Correlation heatmap")
    numeric_df = df_filtered.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
# boxplots:
col3, col4 = st.columns(2)

# boxplot: SNR vs order:
with col3:
    st.subheader("SNR vs Modulator order")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_filtered, x="Order", y="SNR", palette="viridis", ax=ax)
    plt.xlabel("Modulator order")
    plt.ylabel("SNR (dB)")
    plt.title("SNR distribution by modulator order")
    st.pyplot(fig)

# boxplot: SNR vs OSR:
with col4:
    st.subheader("SNR vs OSR")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_filtered, x="OSR", y="SNR", palette="viridis", ax=ax)
    plt.xlabel("OSR")
    plt.ylabel("SNR (dB)")
    plt.title("SNR distribution by OSR")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# SNR distribution: density plot and histogram:
col5, col6 = st.columns(2)

# density plot:
with col5:
    st.subheader("SNR distribution (density)")
    fig, ax = plt.subplots()
    sns.kdeplot(df_filtered["SNR"], fill=True, color="blue", ax=ax)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Density")
    plt.title("SNR density plot")
    st.pyplot(fig)

# histogram:
with col6:
    st.subheader("SNR distribution (histogram)")
    fig = px.histogram(df_filtered, x="SNR", nbins=30, title="SNR Histogram", 
                       labels={"SNR": "SNR (dB)"}, opacity=0.7, color="Levels" if "Levels" in df_filtered.columns else None)
    st.plotly_chart(fig, use_container_width=True)

# architecture comparison:
if "Form" in df_filtered.columns:
    st.subheader("Architecture comparison: SNR vs OSR")
    fig = go.Figure()
    for arch in df_filtered["Form"].unique():
        arch_data = df_filtered[df_filtered["Form"] == arch]
        fig.add_trace(go.Scatter(x=arch_data["OSR"], y=arch_data["SNR"], mode="lines+markers", name=arch))
    fig.update_layout(xaxis_title="OSR", yaxis_title="SNR (dB)", xaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

# SNR vs Hinf gain:
if "Hinf" in df_filtered.columns:
    st.subheader("SNR vs Hinf gain")
    fig = px.scatter(df_filtered, x="Hinf", y="SNR", color="Form" if "Form" in df_filtered.columns else None,
                     labels={"Hinf": "Hinf gain", "SNR": "SNR (dB)"})
    st.plotly_chart(fig, use_container_width=True)

# download filtered data:
st.sidebar.subheader("Download options")
st.sidebar.download_button("Download filtered data", df_filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
