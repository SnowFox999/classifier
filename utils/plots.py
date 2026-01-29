import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(log_dir, save_path=None):
    df = pd.read_csv(f"{log_dir}/metrics.csv")

    # оставляем только epoch-логи
    df = df.dropna(subset=["epoch"])

    # агрегируем по эпохам
    metrics = df.groupby("epoch").agg({
        "train_loss_epoch": "last",
        "val_loss": "last",
        "val_balanced_acc": "last",
    })

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(metrics.index, metrics["train_loss_epoch"], label="train")
    plt.plot(metrics.index, metrics["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics.index, metrics["val_balanced_acc"], label="val_balanced")
    plt.plot(metrics.index, metrics["train_acc"], label="train")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(targets, preds, classes, save_path=None):
    cm = confusion_matrix(targets, preds, normalize="true")

    plt.figure(figsize=(max(14, len(classes) * 0.5),) * 2)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
