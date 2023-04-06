#!/usr/bin/env python3
import argparse
import logging
import multiprocessing

import pandas as pd
import pyarrow as pa
import numpy as np
from parallel_pandas import ParallelPandas
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.DEBUG,
)

IGNORE_INDEX = -100


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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

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

    LOG.info("Done! Converting into an Apache Arrow table...")

    # Convert the DataFrame of the training set into an Apache Arrow table and
    # write out as a file that can be mmapped at training time.
    table = pa.Table.from_pandas(df)

    LOG.info("Writing out tokenized dataset...")
    with pa.OSFile(args.output_file, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    LOG.info("Finished.")


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

    return parser.parse_args()


def _process_training_example(
    tokenizer: PreTrainedTokenizer,
    series: pd.Series,
    append_eos: bool = True,
) -> pd.Series:
    if append_eos:
        series["generation"] += tokenizer.eos_token

    prompt_tokens = tokenizer(series["prompt"],
                              return_tensors="np").input_ids[0]
    response_tokens = tokenizer(series["generation"],
                                return_tensors="np").input_ids[0]
    input_ids = np.concatenate([prompt_tokens, response_tokens], axis=-1)

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
