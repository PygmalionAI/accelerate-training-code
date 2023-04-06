import pathlib
import typing as t
from dataclasses import dataclass, field

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
    }, default=None)


def main() -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        OtherArguments,
        transformers.TrainingArguments,
    ))
    model_args, data_args, other_args, training_args = parser.parse_args_into_dataclasses(
    )

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

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
    )

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
        trainer.save_model()
        trainer.save_state()
        raise ex

    trainer.save_state()


if __name__ == "__main__":
    main()
