import numpy as np
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
sys.path.append("./")

from accelerate import Accelerator
from omegaconf import OmegaConf
from RandAR.utils import instantiate_from_config
from RandAR.model.nlcd_tokenizer import NLCDTokenizer


@dataclass
class Config:
    config_path: str = "configs/randar_nlcd_32.yaml"
    gpt_ckpt: str = "results_nlcd_32/randar_nlcd_32/checkpoints/final"
    data_path: str = "data/data_32.npz"
    device: str = "cpu"
    n_embeddings: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    n_clusters: int = 8
    seed: int = 42
    output_file: str = "results/visualizations/embedding_space_viz.png"
    size: int = 32


def get_model_embeddings(model, dataset, device, n_samples=100, seed=42):
    embeddings = []
    lulc_images = []
    dominant_classes = []
    
    # Randomly shuffle indices for selection
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:n_samples]
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(indices, desc="Extracting embeddings"):
            tokens, label, _ = dataset[i]
            
            lulc_img = tokens.reshape(dataset.data[i].shape)
            lulc_images.append(dataset.data[i])
            
            unique_vals, counts = np.unique(dataset.data[i], return_counts=True)
            dominant_class = unique_vals[np.argmax(counts)]
            dominant_classes.append(dominant_class)
            
            tokens = tokens.unsqueeze(0).to(device)
            cond = torch.tensor([0], dtype=torch.long, device=device)
            
            bs = 1
            token_order = torch.arange(model.block_size, device=device, dtype=torch.long)
            token_order = token_order.unsqueeze(0).repeat(bs, 1)
            
            idx = torch.gather(tokens.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous()
            
            model.freqs_cis = model.freqs_cis.to(device)
            cond_embeddings = model.cls_embedding(cond, train=False)[:, :model.cls_token_num]
            
            token_embeddings = model.tok_embeddings(idx)
            position_instruction_tokens = model.get_position_instruction_tokens(token_order)
            
            from RandAR.model.utils import interleave_tokens
            h = torch.cat(
                (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
                dim=1
            )
            
            token_freqs_cis = model.freqs_cis[model.cls_token_num:].clone().to(token_order.device)[token_order]
            freqs_cis = torch.cat(
                (model.freqs_cis[:model.cls_token_num].unsqueeze(0).repeat(bs, 1, 1, 1), 
                 interleave_tokens(token_freqs_cis, token_freqs_cis)),
                dim=1
            )
            
            input_pos = None
            mask = None
            
            for layer in model.layers:
                if model.grad_checkpointing:
                    h = torch.utils.checkpoint.checkpoint(layer, h, freqs_cis, input_pos, mask, use_reentrant=False)
                else:
                    h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
            
            h = model.norm(h)
            
            h_mean = h[:, model.cls_token_num:].mean(dim=1)
            
            embeddings.append(h_mean.cpu().numpy())
    
    return np.vstack(embeddings), np.array(lulc_images), np.array(dominant_classes)


def nlcd_to_rgb(data):
    vocab_size = int(data.max()) + 1
    return NLCDTokenizer(vocab_size=vocab_size).nlcd_to_rgb(data)


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space of RandAR model")
    parser.add_argument("--config", type=str, default="configs/randar_nlcd_32.yaml")
    parser.add_argument("--gpt-ckpt", type=str, default="results_nlcd_32/randar_nlcd_32/checkpoints/final")
    parser.add_argument("--data-path", type=str, default="data/data_32.npz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-embeddings", type=int, default=100)
    parser.add_argument("--output-file", type=str, default="embedding_space_viz.png")
    args = parser.parse_args()
    
    config = Config(
        config_path=args.config,
        gpt_ckpt=args.gpt_ckpt,
        data_path=args.data_path,
        device=args.device,
        n_embeddings=args.n_embeddings,
        output_file=args.output_file
    )
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print(f"Loading config from {config.config_path}")
    model_config = OmegaConf.load(config.config_path)
    
    print(f"Loading model from {config.gpt_ckpt}")
    model = instantiate_from_config(model_config.ar_model)
    model = model.to(config.device)
    model.eval()
    
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(config.gpt_ckpt, "model.safetensors"))
    model.load_state_dict(state_dict)
    
    print(f"Loading dataset from {config.data_path}")
    from RandAR.dataset.nlcd_dataset import NLCDDataset
    dataset = NLCDDataset(config.data_path, split='train')
    
    print(f"Extracting embeddings for {config.n_embeddings} randomly selected samples...")
    embeddings, lulc_images, dominant_classes = get_model_embeddings(
        model, dataset, config.device, config.n_embeddings, seed=config.seed
    )
    
    print(f"Fitting UMAP to reduce from {embeddings.shape[1]} to 2 dimensions...")
    reducer = UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        n_components=2,
        random_state=config.seed
    )
    embedding_2d = reducer.fit_transform(embeddings)
    
    print(f"Fitting KMeans with {config.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed)
    cluster_labels = kmeans.fit_predict(embedding_2d)
    
    centroid_indices = []
    for i in range(config.n_clusters):
        cluster_mask = cluster_labels == i
        if np.any(cluster_mask):
            cluster_points = embedding_2d[cluster_mask]
            cluster_center = kmeans.cluster_centers_[i]
            
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            closest_idx = np.argmin(distances)
            
            global_idx = np.where(cluster_mask)[0][closest_idx]
            centroid_indices.append(global_idx)
    
    tokenizer = NLCDTokenizer(vocab_size=dataset.vocab_size)
    
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1])
    
    # Main embedding plot in upper left 3x4 area
    ax_main = fig.add_subplot(gs[:3, :4])
    
    # Use the same NLCD colormap for the scatter points
    point_colors = []
    for dom_class in dominant_classes:
        # Map data value to NLCD LUT key
        lut_key = tokenizer.data_to_lut.get(dom_class, 11)  # Default to water if not found
        rgb = tokenizer.lut[lut_key]
        point_colors.append(rgb)
    
    ax_main.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=point_colors, s=15, alpha=0.5)
    
    for idx, centroid_idx in enumerate(centroid_indices[:8]):
        x, y = embedding_2d[centroid_idx]
        # circle = plt.Circle((x, y), 0.04 * (embedding_2d[:, 0].max() - embedding_2d[:, 0].min()), 
        #                    color='black', fill=False, zorder=10)
        # ax_main.add_patch(circle)
        ax_main.text(x, y, str(idx + 1), color='black', fontsize=12, ha='center', va='center', 
                    weight='bold', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='white', alpha=0.5, 
                             edgecolor='none'))

    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # Bottom row (1x5) - first 5 LULC plots
    for i in range(5):
        if i < len(centroid_indices):
            ax = fig.add_subplot(gs[3, i])
            centroid_idx = centroid_indices[i]
            
            img = lulc_images[centroid_idx]
            img_rgb = nlcd_to_rgb(img)
            
            ax.imshow(img_rgb)
            ax.axis('off')
            ax.text(3, 3, str(i + 1), color='black', fontsize=12, ha='left', va='top', 
                    weight='bold',
                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='white', alpha=0.5, 
                             edgecolor='none'))
    
    # Right column (3x1) - remaining 3 LULC plots
    for i in range(3):
        plot_idx = i + 5
        if plot_idx < len(centroid_indices):
            ax = fig.add_subplot(gs[i, 4])
            centroid_idx = centroid_indices[plot_idx]
            
            img = lulc_images[centroid_idx]
            img_rgb = nlcd_to_rgb(img)
            
            ax.imshow(img_rgb)
            ax.axis('off')
            ax.text(3, 3, str(plot_idx + 1), color='black', fontsize=12, ha='left', va='top', 
                    weight='bold',
                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='white', alpha=0.5, 
                             edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(config.output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {config.output_file}")


if __name__ == "__main__":
    main()