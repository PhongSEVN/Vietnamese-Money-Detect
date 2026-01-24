import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision.transforms import Compose, Resize, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize, \
    CenterCrop, RandomRotation, ColorJitter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tqdm
from argparse import ArgumentParser
from config.mobile_train_config import MOBILE_DATA_DIR, MOBILE_EPOCHS, MOBILE_BATCH_SIZE, MOBILE_IMG_SIZE, NUM_WORKERS
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from dataset.dataset import Money
from mobilenet_model import MobileNet


def get_args():
    parser = ArgumentParser(description='MobileNet Training for Banknote Classification')
    parser.add_argument("--epochs", "-e", type=int, default=MOBILE_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=MOBILE_BATCH_SIZE, help="Batch size")
    parser.add_argument("--image_size", "-i", type=int, default=MOBILE_IMG_SIZE, help="Image size")
    parser.add_argument("--root", "-r", type=str, default=MOBILE_DATA_DIR, help="Root of dataset")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard_mobilenet", help="Log training directory")
    parser.add_argument("--model", "-m", type=str, default="trained_mobilenet", help="Model save directory")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Checkpoint to resume training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Normalize confusion matrix (handle division by zero)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_normalized = np.around(cm.astype('float') / row_sums, decimals=2)
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                     horizontalalignment="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure("Confusion Matrix", figure, epoch)
    plt.close(figure)


def calculate_class_weights(dataset):
    labels = dataset.label_path
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = total_samples / (len(class_counts) * class_counts)

    print(f"\nClass Distribution:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"   Class {i} ({dataset.categories[i]}): {count} samples, weight: {weight:.4f}")

    return torch.FloatTensor(class_weights)


def create_model(num_classes, device, freeze_backbone=True):
    model = MobileNet(num_classes=num_classes, freeze_backbone=freeze_backbone)
    return model.to(device)


if __name__ == '__main__':
    args = get_args()

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    if not os.path.exists(args.root):
        print(f"Không tìm thấy thư mục {args.root}. Hãy chuẩn bị data trước!")
        exit(1)

    train_transform = Compose([
        RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(
            brightness=0.3,  # Thay đổi độ sáng
            contrast=0.3,  # Thay đổi độ tương phản
            saturation=0.3,  # Thay đổi độ bão hòa
            hue=0.1  # Thay đổi màu sắc nhẹ
        ),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = Compose([
        Resize(256),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Money(MOBILE_DATA_DIR, train=True, transform=train_transform)
    test_dataset = Money(MOBILE_DATA_DIR, train=False, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = train_dataset.categories
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    class_weights = calculate_class_weights(train_dataset).to(device)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    os.makedirs(args.model, exist_ok=True)

    writer = SummaryWriter(args.logging)

    model = create_model(len(class_names), device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")
    else:
        start_epoch = 0
        best_accuracy = 0

    num_iters = len(train_loader)
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train batches: {num_iters}, Test batches: {len(test_loader)}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, colour='green')

        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(
                f"Epoch [{epoch + 1}/{args.epochs}] | Iter [{iter + 1}/{num_iters}] | Loss: {loss.item():.4f}"
            )
            writer.add_scalar("Train/loss", loss.item(), epoch * num_iters + iter)

        avg_train_loss = running_loss / num_iters
        writer.add_scalar("Train/epoch_loss", avg_train_loss, epoch)

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        # Update learning rate based on accuracy
        scheduler.step(accuracy)
        current_lr = optimizer.param_groups[0]['lr']

        plot_confusion_matrix(writer, cm, class_names, epoch)
        writer.add_scalar("Validation/accuracy", accuracy, epoch)
        writer.add_scalar("Validation/loss", avg_val_loss, epoch)
        writer.add_scalar("Train/learning_rate", current_lr, epoch)

        print(
            f"\nEpoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        # print("\nClassification Report:")
        # print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_val_loss,
            "accuracy": accuracy,
            "class_names": class_names
        }
        torch.save(checkpoint, f"{args.model}/last_mobilenet.pt")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint["best_accuracy"] = best_accuracy
            torch.save(checkpoint, f"{args.model}/best_mobilenet.pt")
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {args.model}/")
    print(f"Tensorboard logs: {args.logging}/")

    writer.close()