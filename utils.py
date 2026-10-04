
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer
from functools import partial


class DishCalorieModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.image_encoder = timm.create_model(
            cfg.IMAGE_MODEL, pretrained=True, num_classes=0
        )
        image_dim = self.image_encoder.num_features

        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL)
        text_dim = self.text_encoder.config.hidden_size

        self.image_projection = nn.Linear(image_dim, cfg.HIDDEN_DIM)
        self.text_projection = nn.Linear(text_dim, cfg.HIDDEN_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.HIDDEN_DIM, 1)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        image_emb = self.image_projection(image_features)

        text_output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        text_emb = self.text_projection(text_features)

        combined = torch.cat([image_emb, text_emb], dim=1)
        output = self.classifier(combined)
        return output.squeeze(1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].float().to(device)

        optimizer.zero_grad()
        pred = model(image, input_ids, attention_mask)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae = torchmetrics.MeanAbsoluteError().to(device)

    for batch in loader:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].float().to(device)

        pred = model(image, input_ids, attention_mask)
        mae.update(pred, target)

    return mae.compute().item()


def train(cfg, train_loader, test_loader):
    torch.manual_seed(cfg.SEED)
    device = cfg.DEVICE

    model = DishCalorieModel(cfg).to(device)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    criterion = nn.L1Loss()

    best_mae = float('inf')

    for epoch in range(cfg.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mae = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | train_loss: {train_loss:.2f} | val_MAE: {val_mae:.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"  Сохранён чекпоинт (MAE={val_mae:.2f})")

    return model, best_mae
