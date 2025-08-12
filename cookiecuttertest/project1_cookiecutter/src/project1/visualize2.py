# --- add these two lines at the very top ---
import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path
from .model import MyAwesomeModel  # you already switched to relative import

DEVICE = torch.device("cpu")  # be conservative for viz

def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    state = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # If you want pre-logit features, your classifier layer is `fc1`
    if hasattr(model, "fc1"):
        model.fc1 = torch.nn.Identity()

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "processed"        # adjust if you use a subfolder
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_images = torch.load(data_dir / "test_images.pt", map_location=DEVICE)
    test_target = torch.load(data_dir / "test_target.pt", map_location=DEVICE)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embs, tgts = [], []
    with torch.inference_mode():
        for images, target in torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0):
            embs.append(model(images.to(DEVICE)).cpu())
            tgts.append(target.cpu())

    import numpy as np
    embeddings = torch.cat(embs).numpy()
    targets = torch.cat(tgts).numpy()

    # Start with PCA->2D only (TSNE often triggers segfaults on mac with some BLAS builds)
    from sklearn.decomposition import PCA
    if embeddings.shape[1] > 2:
        embeddings = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        if np.any(mask):
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], s=12, label=str(i))
    plt.legend()
    out = fig_dir / figure_name
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")

if __name__ == "__main__":
    typer.run(visualize)
