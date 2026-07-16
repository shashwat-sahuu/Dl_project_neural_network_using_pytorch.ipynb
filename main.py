import torch
import torch.nn as nn
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- Recreate your exact model architecture here ---
class BreastCancerNet(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Load model + scaler
model = BreastCancerNet()
model.load_state_dict(torch.load("model (1).pth", map_location="cpu"))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class InputData(BaseModel):
    features: list[float]  # 30 values

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor).item()
    prediction = "Malignant" if output > 0.5 else "Benign"
    return {"prediction": prediction, "probability": round(output, 4)}
