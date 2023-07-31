import math

import numpy as np
import torch

from torch import nn
from transformers import Trainer

IGNORE_DECAY_PARAMS = ["bias", "layer_norm", "layernorm"]

class MeZOOptimizer:
    def __init__(self, trainer: Trainer, model: nn.Module) -> None:
        self.trainer = trainer
        self.batch_size = trainer._train_batch_size
        self.eps = trainer.zo_eps
        self.model = model

        assert trainer.args.gradient_accumulation_steps == 1, "MeZO does not support gradient accumulation!"
        # Initialize named_parameters_to_optim
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

    def step(self, inputs: dict) -> torch.Tensor:
        args = self.trainer.args
        # Reset seeds, gradients and losses
        self.losses = []
        self.projected_grads = []
        self.random_seeds = []

        # Do a step for every entry in the batch
        for i in range(self.batch_size):
            # Fetch input from data, unsqueezing necessary because model wants tensor of shape
            # (batch_size, context_len)
            entry = {
                "input_ids": inputs['input_ids'][i].unsqueeze(0),
                "labels": inputs['labels'][i].unsqueeze(0),
            }
            # Generate seed for sampling z
            seed = np.random.randint(1000000000)

            # First function evaluation
            self._perturb_parameters(random_seed=seed, scaling_factor=1)
            loss1 = self._forward_pass(entry)

            # Second function evaluation
            self._perturb_parameters(random_seed=seed, scaling_factor=-2)
            loss2 = self._forward_pass(entry)

            # Calculate and normalize grad
            self.projected_grad = ((loss1 - loss2) / (2 * self.eps)).item()
            self.projected_grad = self._log_normalize(self.projected_grad)

            # Reset model back to its parameters at start of step
            self._perturb_parameters(random_seed=seed, scaling_factor=1)

            # Append things
            self.losses.append(loss1)
            self.projected_grads.append(self.projected_grad)
            self.random_seeds.append(seed)

        # Return the average of the original losses
        return torch.mean(torch.tensor(self.losses, device=args.device, dtype=torch.float32, requires_grad=False))

    def update(self) -> None:
        '''Updates the parameters.'''
        args = self.trainer.args

        for i in range(self.batch_size):
            # Reset the random seed for sampling zs
            torch.manual_seed(self.random_seeds[i])           

            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = self._sample_z(param)
                if _list_not_in_name(search_terms=IGNORE_DECAY_PARAMS, name=name):
                    param.data = param.data - (self.trainer._get_learning_rate() / self.batch_size) * (self.projected_grads[i] * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - (self.trainer._get_learning_rate() / self.batch_size) * (self.projected_grads[i] * z)
        
        self.trainer.lr_scheduler.step()

    def _forward_pass(self, inputs: dict) -> torch.Tensor:
        self.model.eval()
        
        with torch.inference_mode():
            inputs = self.trainer._prepare_inputs(inputs)
            with self.trainer.compute_loss_context_manager():
                loss = self.trainer.compute_loss(self.model, inputs)
            if self.trainer.args.n_gpu > 1:
                # Warning: this is copied from the original HF Trainer. Untested.
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            
        return loss.detach()

    def _log_normalize(self, num: float) -> float:
        '''
        Normalizes a number via taking the natural log of its absolute value, then reapplying the sign of the original number.
        This is necessary in order for MeZO to converge.
        '''
        if num != 0:
            if num > 0:
                num = math.log(num)
            else:
                num = -1 * math.log(abs(num))
        else:
            num = math.log(self.eps)

        return num

    def _perturb_parameters(self, random_seed: int, scaling_factor: int) -> None:
        '''
        Randomly shifts the parameters to simulate movement in a random direction
        in the dimensional space of the model.
        '''
        torch.manual_seed(random_seed)
        for _, param in self.named_parameters_to_optim:
            # Sample z
            z = self._sample_z(param)
            param.data = param.data + scaling_factor * z * self.eps

    def _sample_z(self, param: nn.Parameter, sign: bool = False) -> torch.Tensor:
        '''Samples a vector z.'''
        z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype
        )
        if sign:
            z = z.sign()
        return z

# Utils

def _list_not_in_name(search_terms: list, name: str) -> bool:
    for t in search_terms:
        if t in name:
            return False
    return True