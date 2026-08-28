Facial Expression Recognition
A deep learning project that recognizes emotions from images using Transfer Learning with EfficientNet-B0.

What It Does
Upload a photo and the model will detect which emotion is shown: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.

How It Works
Model: EfficientNet-B0 (pre-trained on ImageNet, fine-tuned on emotion dataset)
Training: Uses Optuna for hyperparameter tuning (20 trials), then trains on best parameters
Regularization: Dropout, Batch Normalization, L2 weight decay, learning rate scheduling
Backend: FastAPI REST API
Frontend: Streamlit web app
