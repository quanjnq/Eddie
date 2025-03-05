import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.disable(logging.DEBUG)

import matplotlib.pyplot as plt
import numpy as np


def init_fold_plot(k_folds):
    """
    初始化用于绘制 k 个 fold 图的子图网格。
    """
    fig, axes = plt.subplots(5, k_folds, figsize=(4 * k_folds, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    return fig, axes


def update_fold_plot(file_path, fig, axes, fold_idx, train_losses, val_losses, train_labels, train_preds, val_labels, val_preds):
    """
    更新指定 fold 的列图数据。
    """
    # 绘制 loss 曲线
    ax = axes[0, fold_idx]
    ax.plot(list(range(len(train_losses))), train_losses, label='Train Loss', color='red', linestyle=':', marker='.',
            markersize=5)
    ax.plot(list(range(len(val_losses))), val_losses, label='Val Loss', color='green', linestyle=':', marker='.',
            markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title(f"Loss (Fold-{fold_idx + 1})")

    # 绘制训练标签分布
    ax = axes[1, fold_idx]
    ax.hist([l for l in train_labels], bins=50)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Train Labels (Fold-{fold_idx + 1})")

    # 绘制训练预测分布
    ax = axes[2, fold_idx]
    ax.hist([l for l in train_preds], bins=50)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Train Predictions (Fold-{fold_idx + 1})")

    # 绘制验证标签分布
    ax = axes[3, fold_idx]
    ax.hist([l for l in val_labels], bins=50)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Val Labels (Fold-{fold_idx + 1})")

    # 绘制验证预测分布
    ax = axes[4, fold_idx]
    ax.hist([l for l in val_preds], bins=50)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Val Predictions (Fold-{fold_idx + 1})")

    fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    