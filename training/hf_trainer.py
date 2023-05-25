import os
import pathlib
import typing as t
from dataclasses import dataclass, field

import torch
import transformers
import typing
from dataset import DataCollatorForMmapedDataset, MmappedArrowDataset
from profiling import ProfilerCallback, build_profiler_configuration


@dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = field(
        default="EleutherAI/pythia-70m-deduped")
    use_xformers: bool = field(default=False, metadata={"help": "Use xFormers' memory_efficient_attention"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training set."})
    eval_file: str = field(metadata={"help": "Path to the evaluation set."})


@dataclass
class OtherArguments:
    model_load_delay_per_rank: t.Optional[int] = field(metadata={
        "help": "Delay loading the model by (this many seconds) * (local_rank)."},
        default=None)
    enable_profiler: bool = field(
        metadata={"help": "Whether to profile the training loop."},
        default=False)
    add_special_tokens: t.Optional[str] = field(metadata={
        "help": "Extra special tokens to add to the tokenizer before training. Comma-separated."},
        default=None)


@dataclass
class LoraArguments:
    use_lora: t.Optional[bool] = field(metadata={"help": "Whether to train a LoRA instead of the full model."},
                                       default=False)
    lora_rank: t.Optional[int] = field(metadata={"help": "LoRA rank."},
                                       default=4)
    lora_alpha: t.Optional[int] = field(metadata={"help": "LoRA alpha."},
                                        default=32)
    lora_dropout: t.Optional[float] = field(metadata={"help": "LoRA dropout."},
                                            default=0.05)
    lora_target_modules: t.Optional[str] = field(metadata={"help": "Target modules, comma-separated."},
                                                 default=None)

 
@dataclass
class BitsAndBytesArguments:
    load_in_4bit: t.Optional[bool] = field(metadata={
        "help": "Weather to use 4-Bit quantization."},
        default=False)
    load_in_8bit: t.Optional[bool] = field(metadata={
        "help": "Weather to use 8-Bit quantization."},
        default=False)
    quant_type: t.Optional[typing.Literal['nf4', 'fp4']] = field(metadata={
        "help": "Weather to use NF4 quant type. Only used for 4bit."},
        default="fp4")
    use_double_quant: t.Optional[bool] = field(metadata={
        "help": "Weather to use double quant. Only used for 4bit."},
        default=False)


def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        BitsAndBytesArguments,
        OtherArguments,
        transformers.TrainingArguments,
    ))
    model_args, data_args, lora_args, bnb_args, \
        other_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

    # xFormers optimizations.
    if model_args.use_xformers:
        from monkeypatches import apply_xformers_monkeypatches
        apply_xformers_monkeypatches()

    if other_args.model_load_delay_per_rank is not None:
        # When working with constrained system memory, loading the model at the
        # exact same time on all training processes will likely fail due to all
        # the model copies going around. We can delay loading based on
        # local_rank so not all processes are doing this at once, which
        # alleviates the situation. Kinda silly, but it works.
        import time
        time.sleep(other_args.model_load_delay_per_rank *
                   training_args.local_rank)

    # Model loading.
    model_load_dtype = None
    if training_args.bf16:
        model_load_dtype = torch.bfloat16
    elif training_args.fp16:
        model_load_dtype = torch.float16

    if bnb_args.load_in_8bit or bnb_args.load_in_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_8bit=bnb_args.load_in_8bit,
            load_in_4bit=bnb_args.load_in_4bit,
            bnb_4bit_compute_dtype=model_load_dtype,
            bnb_4bit_quant_type=bnb_args.quant_type,
            bnb_4bit_use_double_quant=bnb_args.use_double_quant,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=model_load_dtype,
            quantization_config=quantization_config,
        ).cuda()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=model_load_dtype,
        ).cuda()

    if other_args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./preparation/tokenize_data.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = other_args.add_special_tokens.split(",")
        special_tokens = [
            transformers.AddedToken(
                # Heads up: this is very poorly documented in HuggingFace and
                # some old forum discussions mention that it's apparently
                # exclusive to the Rust-based tokenizers? If anything seems
                # funky about the special token behavior, this is a good place
                # to look.
                content, lstrip=True, rstrip=True)
            for content in special_token_contents
        ]

        _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
            {"additional_special_tokens": special_tokens},
            tokenizer,
            model,
        )

    # LoRA setup.
    if lora_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = None
        if lora_args.lora_target_modules is not None:
            target_modules = lora_args.lora_target_modules.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
        )
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Silence this annoying warning.
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # Dataset setup.
    train_dataset = MmappedArrowDataset(data_args.train_file)
    eval_dataset = MmappedArrowDataset(data_args.eval_file)
    data_collator = DataCollatorForMmapedDataset(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[SavePeftModelCallback] if lora_args.use_lora else None,
    )

    try:
        # Resume from checkpoint if we have any checkpoints automatically saved
        # by the HF Trainer within the output directory.
        resume_from_checkpoint = len(
            list(pathlib.Path(
                training_args.output_dir).glob("checkpoint-*"))) > 0

        if other_args.enable_profiler:
            profiler_args = build_profiler_configuration()
            with torch.profiler.profile(**profiler_args) as profiler:
                trainer.add_callback(ProfilerCallback(profiler=profiler))
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except KeyboardInterrupt as ex:
        # TODO(11b): Test whether this does what I expect. Idea is to have the
        # trainer save the current state when I interrupt the run so I don't
        # need to keep waiting for a checkpoint step.
        # trainer.save_model()
        # trainer.save_state()
        raise ex

    trainer.save_state()
    trainer.save_model()


class SavePeftModelCallback(transformers.TrainerCallback):
    """
    At some point, PEFT stopped saving just the adapter and instead started
    storing full model weights. Extracting the adapter from the weights is
    doable, but seems to result in subpar results for some unknown reason, so
    this Trainer callback saves the adapter itself during training to avoid
    this.

    https://github.com/huggingface/peft/issues/286#issuecomment-1512611968
    https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
    """

    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        checkpoint_folder_name = f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_folder = os.path.join(args.output_dir, checkpoint_folder_name)

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)

        return control


def _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
    special_tokens: t.Dict[str, t.Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    tokenizer.add_special_tokens(special_tokens)

    # Size is rounded up to the nearest number divisible by 64 for performance
    # reasons.
    new_size = _nearest_divisible(num=len(tokenizer), divisor=64)
    old_size = model.config.vocab_size

    if new_size == old_size:
        # No resizing needs to be done, let's bail!
        return

    # Need to resize the token embeddings. We initialize the new positions with
    # the mean of the existing ones to cut down on required training time.
    model.resize_token_embeddings(new_size)
    new_positions_count = new_size - old_size

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # This is just to keep the LSP happy.
    assert isinstance(input_embeddings, torch.Tensor)
    assert isinstance(output_embeddings, torch.Tensor)

    input_embeddings_avg = input_embeddings[:-new_positions_count].mean(dim=0,
                                                             keepdim=True)
    output_embeddings_avg = output_embeddings[:-new_positions_count].mean(dim=0,
                                                               keepdim=True)

    input_embeddings[-new_positions_count:] = input_embeddings_avg
    output_embeddings[-new_positions_count:] = output_embeddings_avg


def _nearest_divisible(num: int, divisor: int) -> int:
    """Returns the nearest number to `num` that is divisible by `divisor`."""
    return (num + divisor - 1) // divisor * divisor


if __name__ == "__main__":
    main()
