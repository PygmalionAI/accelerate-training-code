import pathlib
import typing as t
from dataclasses import dataclass, field

import torch
import transformers
from dataset import DataCollatorForMmapedDataset, MmappedArrowDataset
from profiling import ProfilerCallback, build_profiler_configuration

try:
    import xformers
    # Though we don't use memory_efficient_attention here,
    # xformers being installed wrong can lead to the possibility of it running "import xformers" normally,
    # but fails at "from xformers.ops import memory_efficient_attention" with "no module named xformers.ops".
    # Therefore we make 100% sure xformers is installed correctly by importing memory_efficient_attention directly.
    from xformers.ops import memory_efficient_attention
    XFORMERS_INSTALLED = True
except ImportError:
    XFORMERS_INSTALLED = False


@dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = field(
        default="EleutherAI/pythia-70m-deduped")
    apply_xformers: bool = field(default=False, metadata={"help": "Enable xformers optimizations for attention operations"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training set."})
    eval_file: str = field(metadata={"help": "Path to the evaluation set."})


@dataclass
class OtherArguments:
    model_load_delay_per_rank: t.Optional[int] = field(metadata={
        "help":
        "Delay loading the model by (this many seconds) * (local_rank).",
    },
                                                       default=None)
    enable_profiler: bool = field(
        metadata={"help": "Whether to profile the training loop."},
        default=False)


@dataclass
class LoraArguments:
    use_lora: t.Optional[bool] = field(metadata={"help": "LoRA rank."},
                                       default=False)
    lora_rank: t.Optional[int] = field(metadata={"help": "LoRA rank."},
                                       default=4)
    lora_alpha: t.Optional[int] = field(metadata={"help": "LoRA alpha."},
                                        default=32)
    lora_dropout: t.Optional[float] = field(metadata={"help": "LoRA dropout."},
                                            default=0.05)


def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        OtherArguments,
        transformers.TrainingArguments,
    ))
    model_args, data_args, lora_args, \
        other_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

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

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=model_load_dtype,
    ).cuda()
    
    # Xformers optimization.
    if model_args.apply_xformers:
        assert XFORMERS_INSTALLED, "Xformers is not installed (properly)!"
        from xformers_monkeypatching import apply_xformers_to_model
        apply_xformers_to_model(model)

    # LoRA setup.
    if lora_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
        )
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

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


if __name__ == "__main__":
    main()
