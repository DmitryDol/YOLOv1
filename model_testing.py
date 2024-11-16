import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YOLOLoss
from train import Compose


seed = 123
torch.manual_seed(seed=seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # 64 in original paper 
WEIGHT_DECAY = 2e-4
EPOCHS = 1
# set num workers to the number of available cpu cores
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True
# Need to load pretrained model
LOAD_MODEL = True
# Path to pretrained model
LOAD_MODEL_FILE = "checkpoints/checkpoint_epoch_160.pth.tar"
# Path to images and labels
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
train_dataset_path = 'data/train.csv'
TEST_DATASET_PATH = 'data/test.csv'

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.eval()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=DEVICE), model, optimizer, device='cpu')

    test_dataset = VOCDataset(
        TEST_DATASET_PATH,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    models_mAP = dict()
    
    for epoch in range(EPOCHS):
        
        for x, y in test_loader:
            x = x.to(DEVICE)
            for idx in range(8):
                model_out = model(x)
                bboxes = cellboxes_to_boxes(model_out)
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format='midpoint')
                plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
            input()

        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        
        
    return models_mAP


if __name__ == "__main__":
    models_mAP = main()