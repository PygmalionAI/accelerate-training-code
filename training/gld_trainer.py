"""Gradientless descent."""
# Must modify HF trainer directly to implement GLD
import math
import os
import time
import typing as t
import shutil
import sys

from dataclasses import dataclass
from packaging import version

# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params

import numpy as np
import torch
import torch.distributed as dist

from sklearn.linear_model import LogisticRegressionCV
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from transformers import PreTrainedModel, Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import get_model_param_count, IterableDatasetShard
from transformers.trainer_utils import has_length, HPSearchBackend, ShardedDDPOption, speed_metrics, TrainOutput
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import is_accelerate_available, is_apex_available, is_sagemaker_mp_enabled, is_torch_tpu_available, logging

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


# Adapted from https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py
class GLDTrainer(Trainer):
    def __init__(
        self,
        use_gld,
        min_radius,
        max_radius,
        num_ball_samples,
        **kwargs
    ) -> None:
        self.use_gld = use_gld
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.best_scale = None
        # NOTE(TG): "K" in Algorithm 1 of the paper. What actually is this?
        self.log_radius = math.log(max_radius / min_radius)
        self.search_space = np.linspace(0, self.log_radius, num=num_ball_samples).tolist()

        super().__init__(**kwargs)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ) -> t.Optional[TrainOutput]:
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Get model to optimize
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    # MeZO added: estimate gradient
                    if self.use_gld:
                        tr_loss_step = self.gld_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if self.use_gld:
                        self.gld_update()
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()
                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    ############## Gradientless descent ##############

    def gld_forward(self, model: nn.Module, inputs: t.Any) -> torch.tensor:
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original HF Trainer. Untested.
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            
        return loss.detach()
    
    def gld_step(self, model: nn.Module, inputs: t.Any) -> float:
        """
        Do steps.
        """
        # (loss, seed, scale)
        self.losses: list[tuple[float, int, t.Optional[float]]] = []

        # UNCOMMENT THIS FIRST IF THINGS GO BAD
        self.gld_seed = np.random.randint(1000000000)
        
        # Do one forwards pass without modifying parameters
        loss = self.gld_forward(model=model, inputs=inputs)

        self.losses.append((loss, self.gld_seed, None))
        # Now conduct ball sampling trials
        for radius in self.search_space:
            trial = (2 ** (-1 * radius)) * self.max_radius
            # Obtain the loss for a specific v_k
            loss_metrics = self.do_trial(model=model, inputs=inputs, scale=trial)
            self.losses.append(loss_metrics)

        # Select and return the lowest loss
        lowest_loss = min(self.losses, key=lambda x: x[0])
        loss, self.gld_seed, self.best_scale = lowest_loss

        return loss

    def gld_update(self) -> None:
        # Perturb the parameters with the result which led to the lowest loss
        torch.manual_seed(self.gld_seed)
        # Almost the same as perturbing parameters but adding additional stuff
        for n, p in self.model.named_parameters():
            # NOTE: If things go wrong, do - instead of +
            # Then: add LR, weight decay
            if self.best_scale is not None:
                v_k = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, type=p.data.dtype)
                # TODO(TG): Determine whether I need to add weight decay and as such
                # have this conditional be there. Obviously this conditional
                # doing the same thing no matter what is temporary
                if _list_not_in_name(NO_WD_PARAM_NAMES, n):
                    # Add WD? LR?
                    p.data = p.data + (self.best_scale * v_k)
                else:
                    p.data = p.data + (self.best_scale * v_k)

        self.lr_scheduler.step()

    def do_trial(self, model: nn.Module, inputs: t.Any, scale: float) -> tuple:
        self.gld_seed = np.random.randint(1000000000)
        # Perturb the parameters
        self.perturb_parameters(scale=scale)
        # Get the loss
        loss = self.gld_forward(model, inputs)
        # Return the parameters to before
        self.perturb_parameters(scale=(-1 * scale))

        return loss, self.gld_seed, scale
    
    def perturb_parameters(self, scale: float, seed: t.Optional[int] = None) -> None:
        # UNCOMMENT THIS FIRST IF THINGS GO BAD
        torch.manual_seed(seed if seed is not None else self.gld_seed)

        for _, p in self.named_parameters_to_optim:
            v_k = torch.normal(mean=0, std=1, size=p.data.size(), device=p.data.device, dtype=p.data.dtype)
            p.data = p.data + (scale * v_k) # Add additional EPS argument for stability?

NO_WD_PARAM_NAMES = [
    "bias",
    "layer_norm",
    "layernorm"
]

def _list_not_in_name(search_terms: list, name: str) -> bool:
    for t in search_terms:
        if t in name:
            return False
    return True
