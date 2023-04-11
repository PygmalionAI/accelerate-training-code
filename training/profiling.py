import torch
from transformers import TrainerCallback


class ProfilerCallback(TrainerCallback):
    '''HuggingFace Trainer callback to step the PyTorch profiler properly.'''

    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, *_args, **_kwargs):
        self.profiler.step()


def build_profiler_configuration():
    '''Helper to keep some profiling junk out of the main training code.'''
    return {
        "activities": [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        "schedule": torch.profiler.schedule(
            skip_first=3,
            wait=1,
            warmup=1,
            active=2,
            repeat=2,
        ),
        "on_trace_ready": torch.profiler.tensorboard_trace_handler(
            dir_name="hf-trainer-traces",
        ),
        "profile_memory": False, # Massive overhead, enable manually if needed.
        "with_stack": True,
        "record_shapes": True,
    }
