import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import timm
import os
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

TRAIN_PATH = './data/train'
VAL_PATH = './data/validation'
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = './checkpoints'

EMOTION_CLASSES = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")


def get_transforms():
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_dataloaders(batch_size=32):
    train_transform, val_transform = get_transforms()

    train_dataset = ImageFolder(TRAIN_PATH, transform=train_transform)
    val_dataset = ImageFolder(VAL_PATH, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


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


def multiclass_accuracy(y_pred, y_true):
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    progress_bar = tqdm(train_loader, desc="[TRAIN]")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(images, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = multiclass_accuracy(logits, labels)
        total_acc += acc.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc.item():.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    scheduler.step()

    return avg_loss, avg_acc


def validate_epoch(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    progress_bar = tqdm(val_loader, desc="[VAL]")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            loss, logits = model(images, labels)

            total_loss += loss.item()
            acc = multiclass_accuracy(logits, labels)
            total_acc += acc.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}'
            })

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    return avg_loss, avg_acc


def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.2, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    print(f"\nTrial {trial.number}")
    print(f"Batch Size: {batch_size}, LR: {learning_rate:.2e}, Dropout: {dropout_rate:.2f}, Weight Decay: {weight_decay:.2e}")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = FaceExpressionModel(dropout_rate=dropout_rate, num_classes=7)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(5):
        print(f"Epoch {epoch+1}/5")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, DEVICE)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        best_val_acc = max(best_val_acc, val_acc)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def tune_hyperparameters(n_trials=20):
    print(f"\nStarting Hyperparameter Tuning (n_trials={n_trials})")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner()

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptimization Complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best Hyperparameters:")

    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return best_params


def train_final_model(batch_size=32, learning_rate=0.001, dropout_rate=0.5, weight_decay=0.0001):
    print(f"\nTraining Final Model")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = FaceExpressionModel(dropout_rate=dropout_rate, num_classes=7)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"✅ Best model saved! Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))

    with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\nTraining Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

    return model, history


def load_model(checkpoint_path):
    model = FaceExpressionModel(dropout_rate=0.5, num_classes=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def predict_single_image(image_path, model):
    from PIL import Image

    image = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    emotion = EMOTION_CLASSES[pred_class]
    confidence = probs[0][pred_class].item()
    all_probs = {EMOTION_CLASSES[i]: probs[0][i].item() for i in range(len(EMOTION_CLASSES))}

    return emotion, confidence, all_probs


if __name__ == "__main__":

    print("\n" + "="*50)
    print("Facial Expression Recognition")
    print("="*50)

    print("\nStep 1: Hyperparameter Tuning (20 trials)")
    best_params = tune_hyperparameters(n_trials=20)

    print("\nStep 2: Training Final Model")
    model, history = train_final_model(
        batch_size=int(best_params['batch_size']),
        learning_rate=best_params['lr'],
        dropout_rate=best_params['dropout'],
        weight_decay=best_params['weight_decay']
    )

    print("\n" + "="*50)
    print("Done!")
    print("="*50)
