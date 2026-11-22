import math

import torch

def cosine_with_warmup(optimizer, config):
    
    def scale_for_step(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        
        decay_steps = max(1 ,config.max_steps - config.warmup_steps)

        progress = min(1.0, (step - config.warmup_steps) / decay_steps)
        cosine = 0.5* (1.0 + math.cos(math.pi * progress))
        
        return config.min_lr_ratio + (1.0  -  config.min_lr_ratio)*cosine
 
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, scale_for_step
    )