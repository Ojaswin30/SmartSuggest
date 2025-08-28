import pandas as pd
import torch
import os
import pickle

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "gru_next_app.pt"
APP_IDX_PATH = "app2idx.pkl"  # saved mapping from training
APP_USAGE_LOG = r"C:\Users\IPG 3\AppData\Local\Google\AndroidStudio2024.3.2\device-explorer\samsung SM-E045F\_\data\data\com.example.appusagelogger\files\app_usage_log.txt"
EMOTION_CSV = "emotional_record.csv"
SEQ_LEN = 5  # how many past steps to consider

# ----------------------------
# LOAD MODEL & MAPPING
# ----------------------------
def load_gru_model():
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    return model

def load_app_mapping():
    with open(APP_IDX_PATH, "rb") as f:
        app2idx = pickle.load(f)
    idx2app = {v: k for k, v in app2idx.items()}
    return app2idx, idx2app

# ----------------------------
# EXTRACT LAST SEQUENCE
# ----------------------------
def extract_last_sequence(app_log_path, emotion_csv, seq_len=SEQ_LEN):
    # Load app usage logs
    df = pd.read_csv(app_log_path, header=None)
    df.columns = ["start","end","app","state","duration","camera","audio"]
    df = df[df["app"] != "One UI Home"].tail(seq_len)

    # Normalize duration & count
    df["duration"] = df["duration"].astype(float)
    df["count"] = 1
    max_dur = df["duration"].max() if df["duration"].max() > 0 else 1

    # Load last emotion embedding
    emotion_df = pd.read_csv(emotion_csv)
    if emotion_df.empty:
        last_emotion = torch.zeros(6)
    else:
        last_row = emotion_df.iloc[-1]
        dims = [f"dim_{i}" for i in range(6)]
        last_emotion = torch.tensor([last_row[d] for d in dims], dtype=torch.float32)

    # Build sequence tensor
    features = []
    for _, row in df.iterrows():
        base_feat = torch.tensor([row["duration"]/max_dur, row["count"]], dtype=torch.float32)
        fused_feat = torch.cat([base_feat, last_emotion])
        features.append(fused_feat)

    # Pad if sequence shorter than SEQ_LEN
    while len(features) < seq_len:
        features.insert(0, torch.zeros_like(features[0]))

    seq_tensor = torch.stack(features).unsqueeze(0)  # shape: (1, seq_len, feature_dim)
    return seq_tensor

# ----------------------------
# PREDICT TOP-N APPS
# ----------------------------
def predict_top_n_apps(top_n=3):
    model = load_gru_model()
    app2idx, idx2app = load_app_mapping()
    seq_tensor = extract_last_sequence(APP_USAGE_LOG, EMOTION_CSV)

    with torch.no_grad():
        logits = model(seq_tensor)
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1).squeeze()
        top_vals, top_idx = torch.topk(probs, k=top_n)
        return [(idx2app[idx.item()], val.item()) for idx, val in zip(top_idx, top_vals)]

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    top_predictions = predict_top_n_apps(top_n=3)
    print("Top predicted apps:")
    for app, prob in top_predictions:
        print(f"{app}: {prob:.4f}")
