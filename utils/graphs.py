import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

def plot_federated_training_curve(base_dir):
    csv_path = os.path.join(base_dir, "federated_training_results.csv")
    if not os.path.exists(csv_path):
        return f" No training results found at {csv_path}"

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    round_col = next((c for c in df.columns if "round" in c or "epoch" in c), None)
    acc_col = next((c for c in df.columns if "accuracy" in c or "acc" in c), None)

    if not round_col or not acc_col:
        return "Could not find round/accuracy columns in training CSV."

    fig = px.line(
        df, x=round_col, y=acc_col,
        title="Federated Training Accuracy Curve",
        markers=True,
        labels={round_col: "Training Round", acc_col: "Accuracy"}
    )

    fig.update_layout(template="plotly_white", height=400)
    return fig

def plot_feature_importance(df):
    df = df.sort_values("importance", ascending=False).head(20)
    fig = px.bar(df, x="importance", y="feature", orientation="h", title="Decision Tree Feature Importances")
    fig.update_yaxes(categoryorder="total ascending")
    return fig