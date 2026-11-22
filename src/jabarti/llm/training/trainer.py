import  itertools
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jabarti.llm.training.checkpoint import save_checkpoint
from jabarti.llm.training.schedule import cosine_with_warmup
from jabarti.llm.generation import generate
from jabarti.llm.config import GenerationConfig

def document_cross_entropy(logits,  targets, pad_id):
    
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none"
    ).view(targets.shape)
    
    real = (targets != pad_id).float()
    tokens_per_example = real.sum(dim=1).clamp(min=1.0)
    per_example = (per_token * real).sum(dim=1) / tokens_per_example

    return per_example.mean()  # one scalar for backward()

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        config,
        eval_loader=None,
        tracker=None,
        tokenizer=None,
        device=None
    ):
        if not device:
            self.device= torch.device(
                "cuda"
                if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
            
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        self.tracker = tracker
        self.config = config
        self.tokenizer = tokenizer
        
        
        self.step=0
        
        torch.manual_seed(config.seed)
        
        self.optimizer = self._build_optimizer()
        self.scheduler = cosine_with_warmup(self.optimizer, config)
        self.sample_config = GenerationConfig(max_new_tokens=config.sample_max_new_tokens, seed=config.seed)
        
        
    def _build_optimizer(self):
        
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            
            if parameter.dim() >= 2:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
                
        return torch.optim.AdamW(
            [
                {"params":decay, "weight_decay": self.config.weight_decay},
                {"params":no_decay, "weight_decay":0.0},
            ],
            lr=self.config.learning_rate,
            betas=self.config.betas
        )
        
    def _to_device(self, batch):
        return{key: value.to(self.device) for key, value in batch.items()}
    
    def _batches(self):
        return itertools.chain.from_iterable(
            itertools.repeat(self.train_loader) #DataLoader
        )
        
    
    def train_step(self, batches):
        
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        
        for _ in range(self.config.accumulation_steps):
            batch = self._to_device(next(batches))
            logits, _, _ = self.model(batch["input_ids"])
            loss = document_cross_entropy(logits,batch["targets"], self.model.config.pad_id) / self.config.accumulation_steps
            loss.backward()
            total_loss+= loss.item()
            
        grad_norm = None
        
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )  
        
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss, grad_norm
    
    @torch.no_grad()
    def evaluate(self):
        
        if self.eval_loader is None:
            return None
        
        self.model.eval()
        total, batches = 0.0, 0
        for batch in itertools.islice(self.eval_loader, self.config.eval_batches):
            batch = self._to_device(batch)
            _, loss, _ = self.model(batch["input_ids"], targets=batch["targets"])
            total+=loss.item()
            batches+=1
            
        self.model.train()
        return total /max(1, batches)
    
    def _log(self, metrics):
        
        if self.tracker is not None:
            self.tracker.log(metrics, step=self.step)
            
    def save(self, filename):
        return save_checkpoint(
            f"{self.config.checkpoint_dir}/{filename}",
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step
        )
    
    def train(self):
        config = self.config
        
        started = time.time()
        history = []
        
        print(f"training  on {self.device} for {config.max_steps} steps")
        
        batches = self._batches()
        
        progress = tqdm(
            total=config.max_steps, initial=self.step, unit="step",
            desc=config.run_name, dynamic_ncols=True
        )
        
        while self.step < config.max_steps:
            loss, grad_norm = self.train_step(batches)
            self.step +=1
            learning_rate = self.scheduler.get_last_lr()[0]
            
            progress.update(1)
            progress.set_postfix(loss=f"{loss:.4f}", lr=f"{learning_rate:.2e}")
            
            if self.step %config.log_every == 0:
                metrics = {"train_loss":loss, "learning_rate":learning_rate}
                if grad_norm is  not None:
                     metrics["grad_norm"] = float(grad_norm)
                self._log(metrics)
                history.append({
                    "step":self.step,
                    **metrics
                })
                
            print_every = config.print_every or config.log_every
            if self.step % print_every == 0:
                tqdm.write(
                    f" step {self.step:>6} loss {loss:.4f}"
                    f" lr {learning_rate:.2e}"
                )
                
            if config.eval_every and self.step % config.eval_every ==0:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    self._log(
                        {
                            "eval_loss":eval_loss
                        }
                    )
                    tqdm.write(f" step {self.step:>6}  eval_loss {eval_loss:.4f}")
                    
            if config.save_every and self.step % config.save_every == 0:
                self.save(f"{config.run_name}_step{self.step}.pt")
                
            if config.sample_every and self.step % config.sample_every == 0:
                self._sample()
                
        progress.close()
        elapsed = time.time() - started
        print(f"done: {self.step} steps in {elapsed:.1f}s")
        return history
                
    @torch.no_grad()
    def _sample(self):
        prompts = self.config.sample_prompts
        
        if not prompts or self.tokenizer is None:
            return 
        self.model.eval()  # no dropout while generating
        rows = []
        columns = ["step", "prompt", "completion"]
        
        for i, prompt in enumerate(prompts):
            response = generate(self.model, self.tokenizer, prompt, self.sample_config,
                                device=self.device, chat=self.config.sample_chat)
            
            
            row = [ self.step, prompt, response]
            log_line = f" step {self.step:>6} sample {prompt!r} -> {response!r}"
            
            rows.append(row)
            
            tqdm.write(log_line)
        
        if self.tracker is not None:
            self.tracker.log_table(
                "samples",
                columns=columns, rows=rows, step=self.step
            )
            
        self.model.train()