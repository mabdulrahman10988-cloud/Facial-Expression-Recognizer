from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision import transforms as T
import numpy as np
import timm

app = FastAPI(
    title="Facial Expression Recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './checkpoints/best_model.pth'

EMOTION_CLASSES = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}


class FaceExpressionModel(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=7):
        super(FaceExpressionModel, self).__init__()

        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, labels=None):
        logits = self.model(images)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits


def load_model(checkpoint_path):
    model = FaceExpressionModel(dropout_rate=0.5, num_classes=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded!")
except Exception as e:
    print(f"⚠️ Model not found. Train the model first: python facial_expression.py")
    model = None


@app.get("/")
def read_root():
    return {
        "message": "Facial Expression Recognition API",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_expression(file: UploadFile = File(...)):
    try:
        if model is None:
            return {
                "success": False,
                "error": "Model not loaded. Train the model first."
            }

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_tensor)

        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]

        predicted_class = np.argmax(probs_np)
        predicted_emotion = EMOTION_CLASSES[int(predicted_class)]
        confidence = float(probs_np[predicted_class])

        emotion_scores = {
            EMOTION_CLASSES[i]: float(probs_np[i])
            for i in range(len(EMOTION_CLASSES))
        }

        return {
            "success": True,
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
