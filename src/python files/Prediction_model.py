import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os

# ============================
# Step 1: Aggregate App Usage Logs
# ============================
def preprocess_app_usage(log_path, last_trained_time=None, bin_size="1min"):
    df = pd.read_csv(log_path, header=None)
    df.columns = ["start", "end", "app", "state", "duration", "camera", "audio"]

    # Remove "One UI Home" entries
    df = df[df["app"] != "One UI Home"]
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Filter for incremental training
    if last_trained_time:
        last_trained_dt = pd.to_datetime(last_trained_time)
        df = df[df["start"] > last_trained_dt]

    if df.empty:
        return pd.DataFrame()  # No new logs

    df["bin"] = df["start"].dt.floor(bin_size)
    agg = df.groupby(["bin", "app"]).agg(
        total_duration=("duration", "sum"),
        count=("app", "count")
    ).reset_index()
    return agg

# ============================
# Step 2: Load Emotion Embeddings
# ============================
def load_emotion_embeddings(csv_path):
    emotion_df = pd.read_csv(csv_path)
    emotion_df["timestamp"] = pd.to_datetime(emotion_df["timestamp"])
    dim_cols = [col for col in emotion_df.columns if col.startswith("dim_")]
    emotion_dict = {}
    for _, row in emotion_df.iterrows():
        vec = torch.tensor([float(row[col]) for col in dim_cols], dtype=torch.float32)
        emotion_dict[str(row["timestamp"])] = vec
    return emotion_dict

# ============================
# Step 3: Fuse App Usage + Emotion
# ============================
def fuse_emotions_per_window(agg_df, emotion_dict, emotion_dim=6, add_time_features=True):
    fused_features = []
    for bin_time, group in agg_df.groupby("bin"):
        total_duration = group["total_duration"].sum() / 1000.0  # ms → seconds
        total_count = group["count"].sum()

        ts_str = str(bin_time)
        emotion_vec = emotion_dict.get(ts_str, torch.zeros(emotion_dim, dtype=torch.float32))

        if add_time_features:
            hour_norm = bin_time.hour / 23.0
            minute_norm = bin_time.minute / 59.0
            time_vec = torch.tensor([hour_norm, minute_norm], dtype=torch.float32)
            fused_vec = torch.cat([torch.tensor([total_duration, total_count], dtype=torch.float32),
                                   time_vec, emotion_vec])
        else:
            fused_vec = torch.cat([torch.tensor([total_duration, total_count], dtype=torch.float32),
                                   emotion_vec])

        fused_features.append((str(bin_time), fused_vec))
    return fused_features

# ============================
# Step 4: Build GRU Sequences
# ============================
def build_sequences(fused_features, seq_len=5):
    sequences = []
    labels = []
    feature_dim = fused_features[0][1].shape[0]

    all_apps = list({f[0] for f in fused_features})
    app2idx = {app: i for i, app in enumerate(all_apps)}
    idx2app = {i: app for app, i in app2idx.items()}

    fused_app_names = [f[0] for f in fused_features]
    for i in range(len(fused_features) - seq_len):
        seq_feats = torch.stack([f[1] for f in fused_features[i:i+seq_len]])
        next_app_idx = torch.tensor([app2idx[fused_app_names[i+seq_len]]], dtype=torch.long)
        sequences.append(seq_feats)
        labels.append(next_app_idx)

    return sequences, labels, feature_dim, len(all_apps), app2idx, idx2app

# ============================
# Step 5: GRU Model
# ============================
class NextAppGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_apps):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_apps)

    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out)
        return logits

class AppDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ============================
# Step 6: Train & Save TorchScript Model
# ============================
def train_and_save_model(app_usage_csv, emotion_csv, save_path="gru_next_app.pt",
                         max_epochs=30, patience=5, last_trained_time=None):
    agg_df = preprocess_app_usage(app_usage_csv, last_trained_time)
    if agg_df.empty:
        print("No new logs to train on.")
        return None, None, None

    emotion_dict = load_emotion_embeddings(emotion_csv)
    fused_features = fuse_emotions_per_window(agg_df, emotion_dict)
    sequences, labels, feature_dim, num_apps, app2idx, idx2app = build_sequences(fused_features)

    dataset = AppDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = NextAppGRU(input_dim=feature_dim, hidden_dim=32, num_layers=1, num_apps=num_apps)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    counter = 0

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(x_batch)
            last_logits = logits[:, -1, :]
            y_last = y_batch.squeeze()
            loss = criterion(last_logits, y_last)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            model.eval()
            scripted_model = torch.jit.script(model)
            scripted_model.save(save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        model.train()

    print(f"TorchScript model saved at {save_path}")
    return model, app2idx, idx2app

# ============================
# Step 7: Predict next app (name)
# ============================
def predict_next_app(model, input_sequence, idx2app):
    model.eval()
    with torch.no_grad():
        logits = model(input_sequence.unsqueeze(0))
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        top_idx = torch.argmax(probs, dim=-1).item()
        return idx2app[top_idx]

# ============================
# Step 8: Full vs Incremental Training Orchestration
# ============================
def train_model_with_incremental_check(app_usage_csv, emotion_csv, save_path="gru_next_app.pt",
                                       max_epochs=30, patience=5, last_trained_time=None):
    if not os.path.exists(save_path):
        print("First-time training on all logs...")
        return train_and_save_model(app_usage_csv, emotion_csv, save_path,
                                    max_epochs=max_epochs, patience=patience, last_trained_time=None)
    else:
        print("Incremental retraining on new logs...")
        return train_and_save_model(app_usage_csv, emotion_csv, save_path,
                                    max_epochs=max_epochs, patience=patience, last_trained_time=last_trained_time)

# ============================
# Example Usage
# ============================
if __name__ == "__main__":
    app_usage_csv = "C:\\Users\\IPG 3\\AppData\\Local\\Google\\AndroidStudio2024.3.2\\device-explorer\\samsung SM-E045F\\_\\data\\data\\com.example.appusagelogger\\files\\app_usage_log.txt"
    emotion_csv = "emotional_record.csv"
    save_path = "gru_next_app.pt"
    last_trained_time = "2025-08-28 13:30:00"  # Example timestamp

    model, app2idx, idx2app = train_model_with_incremental_check(app_usage_csv,emotion_csv,save_path,last_trained_time)
    if model:
        print("Model trained and ready. Use predict_next_app() to get app names.")
