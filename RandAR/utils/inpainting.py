import torch
from typing import Tuple
from RandAR.model.generate import sample
import numpy as np


def inpaint_from_arr(
        model,
        img_arr: np.ndarray, mask: np.ndarray, n_samples: int, disallowed_classes: tuple[int] = (0,), cond: int = None, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9, device: str = "cuda") -> np.ndarray:
    '''
    Uses sampling-based approach to produce an infilled version of an image; mask assumes that 1=masked, 0=unmasked
    This is simpler to use because it assumes rectangular images and handles the creation of position tokens and reshaping automatically.

    img_arr: np.ndarray, shape (size, size)
        Image array to inpaint; has discrete values in range 0 - model.vocab_size
    mask: np.ndarray, shape (size, size)
        Mask array to indicate which pixels are masked; 1=masked, 0=unmasked
    n_samples: int
        Number of samples to generate
    disallowed_classes: tuple[int]
        Classes that are not allowed to be generated
    cond: int or None
        Conditional class label to use for inpainting; if None, will use a dummy class
    temperature: float
        Temperature to use for inpainting
    top_k: int
        Top-k to use for inpainting
    top_p: float
        Top-p to use for inpainting
    
    Returns:
    samples_arr: np.ndarray, shape (n_samples, size, size)
    '''
    size = img_arr.shape[-1]

    assert img_arr.shape[-1] == img_arr.shape[-1], "This code assumes that only square images are used."

    real_tokens = img_arr.flatten()
    mask = mask.flatten()
    
    # Get unmasked (visible) positions and tokens
    known_positions = []
    known_tokens = []
    unknown_positions = []

    for pos in range(size*size):
        if mask[pos]:  # Masked region (to be inpainted)
            unknown_positions.append(pos)
        else:  # Unmasked region (visible context)
            known_positions.append(pos)
            known_tokens.append(real_tokens[pos].item())  # Use original token indices 0-model.vocab_size

    assert [tok < model.vocab_size and tok >= 0 for tok in known_tokens]
    
    model.eval()
    
    disallowed_classes = list(disallowed_classes) # Can be used to make certain classes impossible to generate like open water

    samples_arr = np.zeros((n_samples, size, size))


    # This is the label for the class that we want to condition on
    if not cond:
        cond = torch.tensor([0], dtype=torch.long, device=device)  # Dummy class
    else:
        cond = torch.tensor(cond, dtype=torch.long, device=device)
        
    known_positions = torch.tensor(known_positions, dtype=torch.long)
    known_tokens = torch.tensor(known_tokens, dtype=torch.long)
    unknown_positions = torch.tensor(unknown_positions, dtype=torch.long)

    for i in range(n_samples):
        
        with torch.no_grad():
            
            # Change torch seed
            torch.manual_seed(i)
            
            # Use proper inpainting generation
            gen_indices = generate_inpainting(
                model=model,
                cond=cond,
                known_tokens=known_tokens,
                known_positions=known_positions,
                unknown_positions=unknown_positions,
                cfg_scales=[1.0, 1.0],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                disallowed_classes=disallowed_classes
            )
            

        samples_arr[i] = gen_indices[0].cpu().numpy().reshape(size, size)
        
    return samples_arr

def generate_inpainting(model, 
                       cond: torch.Tensor,
                       known_tokens: torch.Tensor,
                       known_positions: torch.Tensor,
                       unknown_positions: torch.Tensor,
                       cfg_scales: Tuple[float, float] = (1.0, 1.0),
                       temperature: float = 1.0,
                       top_k: int = 0,
                       top_p: float = 1.0,
                       disallowed_classes=None,
                       logit_bias=None,
):
    """
    Custom inpainting generation following RandAR approach.
    
    Args:
        model: The RandAR model
        cond: Condition tensor [bs, 1] 
        known_tokens: Known token values [num_known]
        known_positions: Positions of known tokens [num_known]
        unknown_positions: Positions to generate [num_unknown]
        cfg_scales: CFG scale range (start, end)
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Returns:
        torch.Tensor: Generated indices in raster (spatial) order [bs, block_size]
    """

    device = cond.device

    # Logit bias needs to be the of shape (n_classes,)
    if logit_bias is not None:
        logit_bias = logit_bias.to(device)

        if len(logit_bias.shape) == 0:
            logit_bias = logit_bias.unsqueeze(1)

    bs = cond.shape[0]
    
    
    # Combine positions: known first, then unknown (to be generated)
    all_positions = torch.cat([known_positions, unknown_positions]).to(device)
    num_known = len(known_positions)
    num_unknown = len(unknown_positions)
    total_tokens = num_known + num_unknown
    
    # Create token order tensor
    token_order = all_positions.unsqueeze(0).repeat(bs, 1)
    
    # Get position instruction tokens for all positions
    position_instruction_tokens = model.get_position_instruction_tokens(token_order)
    
    # Get frequency encodings for all positions  
    token_freqs_cis = model.freqs_cis[model.cls_token_num:].clone().to(device)[all_positions]
    token_freqs_cis = token_freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1)
    
    # Prepare CFG
    if cfg_scales[-1] > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
        position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
        token_freqs_cis = torch.cat([token_freqs_cis, token_freqs_cis])
        cfg_bs = bs * 2
    else:
        cond_combined = cond  
        cfg_bs = bs
        
    cond_combined_tokens = model.cls_embedding(cond_combined, train=False)
    
    # Setup KV cache
    max_seq_len = cond_combined_tokens.shape[1] + total_tokens * 2
    with torch.device(device):
        model.setup_caches(max_batch_size=cfg_bs, max_seq_length=max_seq_len, dtype=model.tok_embeddings.weight.dtype)
    
    # Convert known tokens to embeddings
    known_token_embeddings = model.tok_embeddings(known_tokens.to(device))  # [num_known, dim]
    if cfg_scales[-1] > 1.0:
        known_token_embeddings = torch.cat([known_token_embeddings, known_token_embeddings])
    
    # Create initial sequence with condition + all known tokens
    # [cls_token, known_pos_0, known_token_0, ..., known_pos_n, known_token_n]
    initial_seq_parts = [cond_combined_tokens]
    initial_freqs_parts = [model.freqs_cis[:model.cls_token_num].unsqueeze(0).repeat(cfg_bs, 1, 1, 1)]
    
    # Add known position-token pairs
    for i in range(num_known):
        # Add position instruction token 
        initial_seq_parts.append(position_instruction_tokens[:, i:i+1])
        initial_freqs_parts.append(token_freqs_cis[:, i:i+1])
        
        # Add known token
        initial_seq_parts.append(known_token_embeddings[i:i+1].unsqueeze(0).repeat(cfg_bs, 1, 1))
        initial_freqs_parts.append(token_freqs_cis[:, i:i+1])
    
    x = torch.cat(initial_seq_parts, dim=1)
    freqs_cis = torch.cat(initial_freqs_parts, dim=1)
    
    # Generate unknown tokens one by one
    generated_tokens = []
    
    for step in range(num_unknown):
        # Add the position instruction token for the next unknown token
        next_pos_token = position_instruction_tokens[:, num_known + step:num_known + step + 1]
        next_pos_freqs = token_freqs_cis[:, num_known + step:num_known + step + 1]
        
        # Create sequence for this step
        step_x = torch.cat([x, next_pos_token], dim=1)
        step_freqs_cis = torch.cat([freqs_cis, next_pos_freqs], dim=1)
        
        # Create input_pos that matches the sequence length
        input_pos = torch.arange(step_x.shape[1], device=device)
        
        # Forward pass
        logits = model.forward_inference(step_x, step_freqs_cis, input_pos)

        if logit_bias is not None:
            logits += logit_bias

        # Apply CFG
        if cfg_scales[-1] > 1.0:
            cfg_scale = cfg_scales[0] + (cfg_scales[1] - cfg_scales[0]) * step / max(1, num_unknown - 1)
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            
        # If disallowed_classes is provided, set the logits of the disallowed classes to -inf. Recall that logits has shape (bs, 1, vocab_size)
        if disallowed_classes is not None:
            logits[..., disallowed_classes] = -float('inf')
        
        # Sample from the last position
        next_token = sample(logits[:, -1:], temperature=temperature, top_k=top_k, top_p=top_p)[0]
        generated_tokens.append(next_token.item())
        
        # Add the generated token to our sequence for the next iteration
        token_embed = model.tok_embeddings(next_token)
        if cfg_scales[-1] > 1.0:
            token_embed = torch.cat([token_embed, token_embed])
        
        x = torch.cat([x, next_pos_token, token_embed], dim=1)
        freqs_cis = torch.cat([freqs_cis, next_pos_freqs, next_pos_freqs], dim=1)
    
    # Combine results in generation order first
    result_indices_gen_order = torch.zeros(total_tokens, dtype=torch.long, device=device)
    result_indices_gen_order[:num_known] = known_tokens
    result_indices_gen_order[num_known:] = torch.tensor(generated_tokens, device=device)
    
    # Convert back to raster order (same as model.generate does)
    result_indices_raster = torch.zeros(model.block_size, dtype=torch.long, device=device)
    result_indices_raster[all_positions] = result_indices_gen_order
    
    model.remove_caches()
    return result_indices_raster.unsqueeze(0)  # Add batch dimension 