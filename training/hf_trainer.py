import pathlib
import typing as t
from dataclasses import dataclass, field

import torch
import transformers
from dataset import DataCollatorForMmapedDataset, MmappedArrowDataset


@dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = field(
        default="EleutherAI/pythia-70m-deduped")


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
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
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
