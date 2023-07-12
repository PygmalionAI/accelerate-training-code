#!/usr/bin/env python3
import argparse
import logging
import multiprocessing

import pandas as pd
import pyarrow as pa
import numpy as np
from parallel_pandas import ParallelPandas
from transformers import AddedToken, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.DEBUG,
)

IGNORE_INDEX = -100

# When appending EOS to the generations, append this many times. Seems helpful
# when training on long (e.g.: many paragraphs/300+ words) examples.
NUM_OF_EOS_TOKENS = 3

def main() -> None:
    args = _parse_args_from_argv()
    cpu_count = multiprocessing.cpu_count()
    LOG.info("Preparing to use %s CPU cores...", cpu_count)
    ParallelPandas.initialize(
        n_cpu=cpu_count,
        split_factor=4,
        disable_pr_bar=False,
    )

    LOG.info("Loading tokenizer...")
    # OpenLLaMA's fast tokenizer is broken on the stable release of transformers.
    # TODO(TG): When newest transformers version which has fixed tokenizer is released,
    # do a version check.
    is_openllama = 'open_llama' in args.tokenizer_path or 'open-llama' in args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=not is_openllama)

    if args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./training/hf_trainer.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = args.add_special_tokens.split(",")
        special_tokens = [
            AddedToken(
                # Heads up: this is very poorly documented in HuggingFace and
                # some old forum discussions mention that it's apparently
                # exclusive to the Rust-based tokenizers? If anything seems
                # funky about the special token behavior, this is a good place
                # to look.
                content, lstrip=True, rstrip=True)
            for content in special_token_contents
        ]

        tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens})

    # Load the entire dataset into memory. Hopefully we won't be working with
    # huge files anytime soon! If this becomes a problem we can use Dask.
    LOG.info("Loading entire dataset into memory...")
    df = pd.read_json(args.input_file, lines=True)
    LOG.info("Done! About to tokenize...")

    # Length warning messes up progress bars, so we silence temporarily.
    # https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(
        logging.ERROR)

    # `executor=threads` drastically slows down the tokenization, but
    # it's a necessary evil. parallel_pandas seems to leak file descriptors when
    # used in multiprocessing mode, and pandarallel deadlocks.
    df = df.p_apply(
        lambda x: _process_training_example(tokenizer, x),
        axis=1,
        executor="threads",
    )

    logging.getLogger("transformers.tokenization_utils_base").setLevel(
        logging.WARNING)

    # Trim out anything bigger than our max length to avoid problems at training
    # time.
    LOG.info("Done! Trimming out any examples longer than %s tokens...",
             args.max_length)

    df = df.loc[df["input_ids"].map(len) <= args.max_length]

    num_tokens = df["input_ids"].map(len).sum()

    LOG.info("Done! Converting into an Apache Arrow table...")

    # Convert the DataFrame of the training set into an Apache Arrow table and
    # write out as a file that can be mmapped at training time.
    table = pa.Table.from_pandas(df)
    schema = pa.schema([
        pa.field('input_ids', pa.list_(pa.int32())),
        pa.field('labels', pa.list_(pa.int32()))
    ])

    LOG.info("Writing out tokenized dataset...")
    with pa.OSFile(args.output_file, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, schema) as writer:
            #writer.write_table(table)
            for row in df.itertuples(index=False):
                # Create a RecordBatch with a singular entry.
                batch = pa.RecordBatch.from_pylist([{"input_ids": row.input_ids, "labels": row.labels}])
                writer.write_batch(batch)


    LOG.info(f"Done! Output file saved to {args.output_file}.")
    LOG.info(f"Dataset contains {num_tokens:,} tokens.")


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset tokenizer utility.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the output binarized and tokenized file.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer-path",
        required=True,
        help="Path to the HF tokenizer to use.",
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        default=2048,
        help=
        "Max length in tokens before a training example is discarded. Defaults to 2048.",
    )
    parser.add_argument(
        "-s",
        "--add-special-tokens",
        type=str,
        default=None,
        help="Extra special tokens to add to the tokenizer before tokenizing. Comma-separated."
    )

    return parser.parse_args()


def _process_training_example(
    tokenizer: PreTrainedTokenizer,
    series: pd.Series,
    append_eos: bool = True,
) -> pd.Series:
    # This is a single row so we _theoretically_ don't have to do this, but if
    # we don't we get a scary warning.
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    generation = series.loc["generation"]

    # TODO(11b): Do a more robust check here.
    is_llama = tokenizer.eos_token == "</s>"

    if append_eos:
        for _ in range(NUM_OF_EOS_TOKENS):
            # As it turns out, with LLaMA's tokenizer, if you just append EOS to
            # the end of the text it gets tokenized as a "</s>" text literal,
            # and not as token #2 which is the _actual_ EOS token. You must have
            # a space before "</s>" so it becomes token #2 as expected.
            #
            # I found this out after wasting 60+ GPU hours training a broken
            # model :)
            if is_llama:
                generation += f" {tokenizer.eos_token}"
            else:
                generation += tokenizer.eos_token

    prompt_tokens = tokenizer(series["prompt"],
                              return_tensors="np").input_ids[0]

    # The LLaMA tokenizer will add a BOS token whenever you tokenize
    # something by default. If we allow this to happen in the response segment,
    # it will cause wildly inconsistent behaviors where the model itself will
    # learn to output BOS in the middle of sentences depending on how input
    # tokenization is done. Not great, so we just force-disable BOS on the
    # response segment.
    response_tokenizer_kwargs = {"add_special_tokens": False} if is_llama else {}
    response_tokens = tokenizer(generation,
                                return_tensors="np",
                                **response_tokenizer_kwargs).input_ids[0]

    input_ids = np.concatenate([prompt_tokens, response_tokens], axis=-1)

    # Let's not waste any more GPU time thanks to this.
    if append_eos:
        assert input_ids[-1].item() == tokenizer.eos_token_id, \
            "EOS was not correctly appended to the end of the response tokens."

    prompt_length = prompt_tokens.shape[-1]
    labels = np.concatenate([
        np.full((prompt_length), IGNORE_INDEX),
        response_tokens,
    ],
                            axis=-1)

    return pd.Series({
        "input_ids": input_ids,
        "labels": labels,
    })

if __name__ == "__main__":
    main()
