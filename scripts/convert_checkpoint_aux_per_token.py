from dataclasses import dataclass
import argparse
import os
import shutil
import sys
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

# Make local imports work when running as a script
sys.path.append("./")

from RandAR.utils import instantiate_from_config

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception as e:
    raise RuntimeError("safetensors is required for converting checkpoints. Please install with `pip install safetensors`.\n" + str(e))


@dataclass
class ConvertConfig:
    ckpt_dir: str
    out_dir: Optional[str]
    config_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    strict: bool = False
    dummy_forward: bool = True


def load_state_dict_safetensors(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model.safetensors not found at {path}")
    state_dict = safe_load_file(path, device="cpu")
    return state_dict


def save_state_dict_safetensors(state_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure all tensors are on CPU for saving
    cpu_state = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in state_dict.items()}
    safe_save_file(cpu_state, path)


def convert_checkpoint(cfg: ConvertConfig) -> Tuple[list, list]:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.out_dir is None:
        cfg.out_dir = cfg.ckpt_dir.rstrip("/") + "_converted"
    os.makedirs(cfg.out_dir, exist_ok=True)

    model_weights_path = os.path.join(cfg.ckpt_dir, "model.safetensors")
    print(f"Loading weights from: {model_weights_path}")
    state_dict = load_state_dict_safetensors(model_weights_path)

    # Load model from provided config
    exp_conf = OmegaConf.load(cfg.config_path)
    # Ensure vocab_size is set if dataset/tokenizer modify it at runtime
    if hasattr(exp_conf, "ar_model") and hasattr(exp_conf.ar_model, "params"):
        # No-op; assume config is correct for the checkpoint
        pass

    model = instantiate_from_config(exp_conf.ar_model).to(cfg.device)
    model.eval()

    # Load state dict (allow non-strict for minor deltas)
    missing, unexpected = model.load_state_dict(state_dict, strict=cfg.strict)
    if missing:
        print("Missing keys (not loaded into model):")
        for k in missing:
            print("  ", k)
    if unexpected:
        print("Unexpected keys (present in checkpoint, not in model):")
        for k in unexpected:
            print("  ", k)

    # Optional: dummy forward to validate shapes
    if cfg.dummy_forward:
        with torch.no_grad():
            block_size = getattr(model, "block_size", None)
            vocab_size = getattr(model, "vocab_size", None)
            num_classes = getattr(model, "num_classes", None)
            aux_dim = getattr(model, "aux_dim", None)
            if block_size is None or vocab_size is None or num_classes is None:
                print("Warning: model missing expected attributes; skipping dummy forward.")
            else:
                bsz = 2
                x = torch.randint(0, vocab_size, (bsz, block_size), device=cfg.device)
                cond = torch.randint(0, num_classes, (bsz,), device=cfg.device)
                aux = None
                if aux_dim is not None and aux_dim > 0:
                    aux = torch.zeros(bsz, block_size, aux_dim, device=cfg.device)
                logits, loss, token_order = model(x, cond, aux=aux, targets=x)
                print("Dummy forward ok:", logits.shape, loss.item())

    # Write out a new checkpoint directory by copying over ancillary files and replacing model.safetensors
    for fname in os.listdir(cfg.ckpt_dir):
        src = os.path.join(cfg.ckpt_dir, fname)
        dst = os.path.join(cfg.out_dir, fname)
        if fname == "model.safetensors":
            continue  # will overwrite below
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    out_model_path = os.path.join(cfg.out_dir, "model.safetensors")
    print(f"Saving converted weights to: {out_model_path}")
    save_state_dict_safetensors(model.state_dict(), out_model_path)

    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Convert old RandAR checkpoint to per-token-aux compatible model.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Path to old checkpoint directory (contains model.safetensors)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used to instantiate the model")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for converted checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to load model on (cuda|cpu)")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading state dict")
    parser.add_argument("--no-dummy-forward", action="store_true", help="Skip dummy forward validation")
    args = parser.parse_args()

    cfg = ConvertConfig(
        ckpt_dir=args.ckpt_dir,
        out_dir=args.out_dir,
        config_path=args.config,
        device=(args.device or ("cuda" if torch.cuda.is_available() else "cpu")),
        strict=args.strict,
        dummy_forward=(not args.no_dummy_forward),
    )

    missing, unexpected = convert_checkpoint(cfg)
    if missing or unexpected:
        print("Conversion finished with key differences. Inspect logs above.")
    else:
        print("Conversion finished with no key differences.")


if __name__ == "__main__":
    main() 