
class Config:
    DATA_DIR = "data/data"
    DRIVE_DIR = "/content/drive/MyDrive/calorie_project"
    SEED = 42

    IMAGE_SIZE = 224
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    EPOCHS = 10
    IMAGE_LR = 1e-4
    TEXT_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4

    HIDDEN_DIM = 512
    IMAGE_MODEL = "efficientnet_b0"
    TEXT_MODEL = "cointegrated/rubert-tiny2"
    MAX_LEN = 64

    DEVICE = "cuda"
    SAVE_PATH = "best_model.pt"

cfg = Config()
