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


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

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
