#!/usr/bin/env python3
"""
IBM Deep Learning Capstone -- Dashboard Application

Streamlit dashboard for exploring deep learning training results,
visualizing metrics, and demonstrating model performance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime


def generate_training_metrics(epochs: int = 50) -> pd.DataFrame:
    """Generate simulated training metrics for dashboard demonstration."""
    np.random.seed(42)
    train_loss = 2.5 * np.exp(-0.06 * np.arange(epochs)) + np.random.normal(0, 0.03, epochs)
    val_loss = 2.5 * np.exp(-0.05 * np.arange(epochs)) + np.random.normal(0, 0.05, epochs)
    train_acc = 1.0 - train_loss / 3.0 + np.random.normal(0, 0.01, epochs)
    val_acc = 1.0 - val_loss / 3.0 + np.random.normal(0, 0.02, epochs)

    return pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Train Loss": np.clip(train_loss, 0.01, None),
        "Val Loss": np.clip(val_loss, 0.01, None),
        "Train Accuracy": np.clip(train_acc, 0, 1),
        "Val Accuracy": np.clip(val_acc, 0, 1),
    })


def generate_confusion_matrix(num_classes: int = 10) -> np.ndarray:
    """Generate a simulated confusion matrix."""
    np.random.seed(42)
    cm = np.random.randint(0, 5, (num_classes, num_classes))
    np.fill_diagonal(cm, np.random.randint(40, 60, num_classes))
    return cm


def main():
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="Deep Learning Capstone", layout="wide")
    st.title("IBM Deep Learning Capstone")
    st.markdown("Dashboard for CNN training metrics and model analysis")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Parameters")
    epochs = st.sidebar.slider("Epochs", 10, 100, 50)
    num_classes = st.sidebar.slider("Classes", 2, 20, 10)
    class_names = [f"Class {i}" for i in range(num_classes)]

    # Generate data
    metrics = generate_training_metrics(epochs)
    cm = generate_confusion_matrix(num_classes)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    best_epoch = int(metrics["Val Accuracy"].idxmax()) + 1
    with col1:
        st.metric("Best Val Accuracy", f"{metrics['Val Accuracy'].max():.2%}")
    with col2:
        st.metric("Final Train Loss", f"{metrics['Train Loss'].iloc[-1]:.4f}")
    with col3:
        st.metric("Best Epoch", best_epoch)
    with col4:
        st.metric("Total Epochs", epochs)

    # Training curves
    st.subheader("Training Curves")
    col_l, col_r = st.columns(2)

    with col_l:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=metrics["Epoch"], y=metrics["Train Loss"],
            name="Train Loss", line=dict(color="#1f77b4")
        ))
        fig_loss.add_trace(go.Scatter(
            x=metrics["Epoch"], y=metrics["Val Loss"],
            name="Val Loss", line=dict(color="#ff7f0e", dash="dash")
        ))
        fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_r:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=metrics["Epoch"], y=metrics["Train Accuracy"],
            name="Train Accuracy", line=dict(color="#2ca02c")
        ))
        fig_acc.add_trace(go.Scatter(
            x=metrics["Epoch"], y=metrics["Val Accuracy"],
            name="Val Accuracy", line=dict(color="#d62728", dash="dash")
        ))
        fig_acc.update_layout(
            title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig_cm = px.imshow(
        cm, labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names, y=class_names,
        color_continuous_scale="Blues", text_auto=True
    )
    fig_cm.update_layout(width=700, height=600)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class metrics
    st.subheader("Per-Class Performance")
    precision = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
    recall = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    class_df = pd.DataFrame({
        "Class": class_names,
        "Precision": np.round(precision, 3),
        "Recall": np.round(recall, 3),
        "F1-Score": np.round(f1, 3),
    })
    st.dataframe(class_df, use_container_width=True)

    # Raw data
    with st.expander("Training History (Raw Data)"):
        st.dataframe(metrics, use_container_width=True)

    st.success("Dashboard loaded successfully.")


if __name__ == "__main__":
    main()
