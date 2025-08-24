import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("./")
from omegaconf import OmegaConf
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
import logging
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

from RandAR.utils import instantiate_from_config
from RandAR.utils.visualization import make_grid
from RandAR.utils.lr_scheduler import get_scheduler

TOKENIZE = True

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data


def main(args):
    config = OmegaConf.load(args.config)
    
    # Setup accelerator
    experiment_dir = os.path.join(config.results_dir, config.exp_name)
    accelerator_config = ProjectConfiguration(project_dir=experiment_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.accelerator.mixed_precision,
        log_with=None if args.no_wandb else config.accelerator.log_with,
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
    )
    set_seed(config.global_seed + accelerator.process_index)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Create a simple logger without using dist.get_rank()
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting NLCD training")
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        
    logger.info(f"Using device {accelerator.device} with {accelerator.num_processes} processes")


    # Create dataset and get value mapping
    dataset = instantiate_from_config(config.dataset)
    value_mapping = dataset.idx_to_value  # Get the value mapping from dataset
    vocab_size = dataset.vocab_size
    
    # Update config with actual vocab size
    config.ar_model.params.vocab_size = vocab_size
    config.tokenizer.params.vocab_size = vocab_size
    
    per_gpu_batch_size = int(
        config.global_batch_size // accelerator.num_processes // config.accelerator.gradient_accumulation_steps
    )
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    logger.info(f"Dataset contains {len(dataset)} samples, batch size: {per_gpu_batch_size}")
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create model
    model = instantiate_from_config(config.ar_model).to(accelerator.device)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create tokenizer with value mapping (not an nn.Module, so no parameters)
    tokenizer = instantiate_from_config(config.tokenizer)
    tokenizer.idx_to_value = value_mapping  # Pass the mapping to tokenizer
    tokenizer.value_to_idx = {v: k for k, v in value_mapping.items()}

    # Create optimizer and scheduler
    optimizer = model.configure_optimizer(**config.optimizer)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=config.max_iters * config.accelerator.gradient_accumulation_steps * accelerator.num_processes,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )
    
    model.train()
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(model, optimizer, data_loader, lr_scheduler)
    data_loader = cycle(data_loader)

    # Resume from checkpoint if specified
    train_steps = 0
    if hasattr(config, 'resume_from') and config.resume_from is not None:
        if os.path.exists(config.resume_from):
            if accelerator.is_main_process:
                logger.info(f"Resuming from checkpoint: {config.resume_from}")
            accelerator.load_state(config.resume_from)
            if accelerator.is_main_process:
                logger.info(f"Successfully loaded checkpoint from {config.resume_from}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Resume checkpoint not found at {config.resume_from}, starting from scratch")
            train_steps = 0

    # Initialize wandb if not disabled
    if not args.no_wandb and accelerator.is_main_process:
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        if config.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        accelerator.init_trackers(
            project_name="RandAR-NLCD",
            init_kwargs={
                "wandb": {
                    "entity": config.wandb_entity,
                    "config": dict(config),
                    "name": config.exp_name,
                    "dir": experiment_dir,
                }
            },
        )
    
    # Training loop
    running_loss = 0
    start_time = time.time()
    
    logger.info(f"Starting training for {config.max_iters} iterations")
    
    # Optional profiling
    profiler = None
    if args.profile and train_steps == 0:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=50, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(experiment_dir, 'profiler')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    while train_steps < config.max_iters:
        model.train()
        x, y, _ = next(data_loader)
        x, y = x.to(accelerator.device, non_blocking=True), y.to(accelerator.device, non_blocking=True)
        image_tokens = x  # Already flattened in dataset
        cond = y.reshape(-1)

        with accelerator.accumulate(model):
            logits, loss, token_order = model(image_tokens, cond, targets=image_tokens)
            accelerator.backward(loss)
            
            if accelerator.sync_gradients and config.optimizer.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        running_loss += accelerator.gather(loss.repeat(per_gpu_batch_size)).mean().item() / config.accelerator.gradient_accumulation_steps

        if accelerator.sync_gradients:
            train_steps += 1
            
            # Step profiler
            if profiler is not None:
                profiler.step()
                if train_steps == 70:  # Stop profiling after warmup + active steps
                    profiler.stop()
                    logger.info("Profiling completed. Results saved to profiler directory.")
                    profiler = None
            
            # Logging
            if train_steps % config.log_every == 0 and accelerator.is_main_process:
                avg_loss = running_loss / config.log_every
                elapsed = time.time() - start_time
                samples_per_sec = (config.global_batch_size * config.log_every) / elapsed
                logger.info(f"Step {train_steps:05d} | Loss {avg_loss:.4f} | Time {elapsed:.2f}s | LR {lr_scheduler.get_last_lr()[0]:.5f} | Samples/sec: {samples_per_sec:.1f}")
                
                if not args.no_wandb:
                    accelerator.log({
                        "loss": avg_loss, 
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples_per_second": samples_per_sec,
                    }, step=train_steps)
                
                running_loss = 0
                start_time = time.time()
            
            # Visualization
            if train_steps % config.visualize_every == 0 and accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    vis_num = min(16, x.shape[0])
                    
                    # Teacher forcing reconstruction
                    pred_indices = torch.argmax(logits[:vis_num], dim=-1)
                    orig_token_order = torch.argsort(token_order[:vis_num])
                    pred_indices_ordered = torch.gather(pred_indices, 1, orig_token_order)
                    pred_imgs = tokenizer.decode_codes_to_img(pred_indices_ordered.detach().cpu(), decode_table=dataset.decode_table)

                    # Ground truth
                    gt_imgs = tokenizer.decode_codes_to_img(image_tokens[:vis_num].detach().cpu(), decode_table=dataset.decode_table)
                        
                    # Generation
                    # Handle both single and multi-GPU cases
                    if hasattr(model, 'module'):
                        gen_indices = model.module.generate(
                            cond=cond[:vis_num],
                            token_order=None,
                            cfg_scales=[1.0, 1.0],
                            num_inference_steps=-1,
                            temperature=1.0,
                            top_k=0,
                            top_p=1.0,
                        )
                        model.module.remove_caches()
                    else:
                        gen_indices = model.generate(
                            cond=cond[:vis_num],
                            token_order=None,
                            cfg_scales=[1.0, 1.0],
                            num_inference_steps=-1,
                            temperature=1.0,
                            top_k=0,
                            top_p=1.0,
                        )
                        model.remove_caches()
                    gen_imgs = tokenizer.decode_codes_to_img(gen_indices.detach().cpu() if isinstance(gen_indices, torch.Tensor) else gen_indices, decode_table=dataset.decode_table)

                    # Save visualizations
                    vis_dir = os.path.join(experiment_dir, "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    pred_grid = make_grid(pred_imgs)
                    gt_grid = make_grid(gt_imgs)
                    gen_grid = make_grid(gen_imgs)
                    
                    Image.fromarray(pred_grid).save(os.path.join(vis_dir, f"pred_{train_steps:05d}.png"))
                    Image.fromarray(gt_grid).save(os.path.join(vis_dir, f"gt_{train_steps:05d}.png"))
                    Image.fromarray(gen_grid).save(os.path.join(vis_dir, f"gen_{train_steps:05d}.png"))
                    
                    logger.info(f"Saved visualizations at step {train_steps}")
                    
                    if not args.no_wandb:
                        import wandb
                        accelerator.log({
                            "pred_recon": wandb.Image(pred_grid),
                            "gt": wandb.Image(gt_grid),
                            "generation": wandb.Image(gen_grid),
                        }, step=train_steps)
                
                model.train()
            
            # Checkpointing
            if train_steps % config.ckpt_every == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(checkpoint_dir, f"iter_{train_steps:05d}")
                os.makedirs(ckpt_path, exist_ok=True)
                accelerator.save_state(ckpt_path)
                logger.info(f"Saved checkpoint at iteration {train_steps}")
    
    # Final checkpoint
    if accelerator.is_main_process:
        final_ckpt_path = os.path.join(checkpoint_dir, "final")
        os.makedirs(final_ckpt_path, exist_ok=True)
        accelerator.save_state(final_ckpt_path)
        logger.info("Training completed!")
    
    accelerator.wait_for_everyone()
    if not args.no_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nlcd/randar_nlcd.yaml")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()
    main(args) 
    