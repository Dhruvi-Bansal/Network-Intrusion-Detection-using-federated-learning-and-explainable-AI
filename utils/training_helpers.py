import os
import pandas as pd
import plotly.express as px

def plot_federated_training_curve(base_dir):
    csv_path = os.path.join(base_dir, "federated_training_results.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    round_col = next((c for c in df.columns if "round" in c or "epoch" in c), None)
    acc_col = next((c for c in df.columns if "accuracy" in c or "acc" in c), None)

    if not round_col or not acc_col:
        return None

    fig = px.line(
        df, x=round_col, y=acc_col,
        title="Training Accuracy Over Rounds",
        markers=True,
        labels={round_col: "Round", acc_col: "Accuracy"}
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig