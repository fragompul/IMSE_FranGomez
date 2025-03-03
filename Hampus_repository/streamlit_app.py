import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset:
df = pd.read_csv("sigma_delta_results.csv")

# Streamlit page configuration:
st.set_page_config(page_title="SDM analysis", layout="wide")

# title:
st.title("Sigma-Delta Modulator analysis")

# sidebar filters:
st.sidebar.header("Filter options")

# select modulator order:
order = st.sidebar.selectbox("Select modulator order:", sorted(df["Order"].unique()))

# select architecture:
if "Form" in df.columns:
    architecture = st.sidebar.selectbox("Select architecture:", sorted(df["Form"].unique()))
    df = df[df["Form"] == architecture]

# OSR range slider:
osr_min, osr_max = float(df["OSR"].min()), float(df["OSR"].max())
osr_range = st.sidebar.slider("Select OSR range:", osr_min, osr_max, (osr_min, osr_max))

# select Hinf gain:
if "Hinf" in df.columns:
    hinf_range = st.sidebar.slider("Select Hinf range:", float(df["Hinf"].min()), float(df["Hinf"].max()), 
                                   (float(df["Hinf"].min()), float(df["Hinf"].max())))

# select quantization levels:
if "Levels" in df.columns:
    levels = st.sidebar.multiselect("Select quantization levels:", sorted(df["Levels"].unique()), default=df["Levels"].unique())

# apply filters:
df_filtered = df[(df["Order"] == order) & (df["OSR"].between(*osr_range))]

if "Hinf" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Hinf"].between(*hinf_range)]

if "Levels" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Levels"].isin(levels)]

# display filtered data:
st.write(f"Filtered data: {len(df_filtered)} entries")
st.dataframe(df_filtered)

# charts:
col1, col2 = st.columns(2)

# SNR vs OSR:
with col1:
    st.subheader("Effect of OSR on SNR")
    fig, ax = plt.subplots()
    sns.lineplot(data=df_filtered, x="OSR", y="SNR", marker="o", ax=ax)
    plt.xscale("log")
    plt.xlabel("OSR")
    plt.ylabel("SNR (dB)")
    plt.title(f"SNR vs OSR (order {order})")
    st.pyplot(fig)

# correlation heatmap:
with col2:
    st.subheader("Correlation heatmap")
    numeric_df = df_filtered.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# additional plots:
col3, col4 = st.columns(2)

# scatter plot: OSR vs SNR:
with col3:
    st.subheader("Scatter plot: OSR vs SNR")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_filtered, x="OSR", y="SNR", hue="Levels", palette="viridis", ax=ax)
    plt.xscale("log")
    plt.xlabel("OSR")
    plt.ylabel("SNR (dB)")
    plt.title(f"SNR vs OSR (order {order})")
    st.pyplot(fig)

# density plot: SNR distribution:
with col4:
    st.subheader("SNR density plot")
    fig, ax = plt.subplots()
    sns.kdeplot(df_filtered["SNR"], fill=True, color="blue", ax=ax)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Density")
    plt.title("SNR distribution")
    st.pyplot(fig)

# download filtered data:
st.sidebar.subheader("Download options")
st.sidebar.download_button("Download filtered data", df_filtered.to_csv(index=False), "filtered_data.csv", "text/csv")
