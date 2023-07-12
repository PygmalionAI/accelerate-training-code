import argparse
import logging
import os

import numpy as np
import pyarrow as pa

from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizer

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.DEBUG,
)

def main() -> None:
    args = _parse_args_from_argv()
    assert os.path.isfile(args.input_path) or os.path.isdir(args.input_path), f'File or directory \"{args.input_path}\" not found!'

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
        
    # Check if it's a directory of .txt files or a specific file
    LOG.info("Done! About to tokenize file(s)...")

    if os.path.isfile(args.input_path):
        all_file_tokens, total_num_tokens = _tokenize_file(tokenizer, args.input_path, args.max_length)
    # Runs this if and only if args.input_path is a directory
    else:
        all_file_tokens: list[np.array] = []
        total_num_tokens = 0
        # Find all .txt files from a directory which could potentially
        # contain other files.
        txt_files = filter(lambda x: x.endswith(".txt"), os.listdir(args.input_path))
        txt_files = [os.path.join(args.input_path, f) for f in txt_files]

        # Obtain the list of token arrays
        for file in txt_files:
            file_tokens, num_tokens = _tokenize_file(tokenizer, file, args.max_length)
            all_file_tokens += file_tokens
            total_num_tokens += num_tokens

    _save_as_arrow_file(all_file_tokens, args.output_file)
    LOG.info(f"Done! Output file saved to {args.output_file}.")
    LOG.info(f"Dataset contains {total_num_tokens:,} tokens.")

def _parse_args_from_argv() -> argparse.Namespace:
    '''Parses arguments.'''
    parser = argparse.ArgumentParser(description="Dataset tokenizer utility.")
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path to the input .txt file or folder containing .txt files.",
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
        "-s",
        "--add-special-tokens",
        type=str,
        default=None,
        help="Extra special tokens to add to the tokenizer before tokenizing. Comma-separated."
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        default=2048,
        help=
        "The maximum amount of tokens the model will take in a batch.\
        Splits the tokenized dataset into chunks of equal length unless the total amount of tokens does not factor cleanly into the context length.\
        In that case, an extra chunk will be added in with the remaining tokens.\
        Defaults to 2048.",
    )

    return parser.parse_args()

def _tokenize_file(tokenizer: PreTrainedTokenizer, filepath: str, max_length: int, append_eos: bool = True) -> tuple[list[np.array], int]:
    '''
    Opens a singular text document and converts its contents into a large array of tokens.

    Params:
    tokenizer: The specific tokenizer used to tokenize the file.
    filepath: The path to the text document that will be tokenized.
    '''
    LOG.info(f"Loading file {filepath} into memory and tokenizing...")

    is_llama = tokenizer.eos_token == "</s>"
    
    with open(filepath, "r", encoding="utf-8") as f:
        # Read the entire .txt file into memory.
        # Good luck!
        file_contents = f.read()
        if append_eos:
            if is_llama:
                file_contents += f" {tokenizer.eos_token}"
            else:
                file_contents += tokenizer.eos_token

        tokenized_contents = tokenizer(file_contents, return_tensors="np").input_ids[0]

    num_tokens = len(tokenized_contents)

    # Do some list slicing to capture chunks of `context_length` tokens...
    closest_ctxlen_factor = (num_tokens // max_length) * max_length
    splitable_tkn_chunks = tokenized_contents[:closest_ctxlen_factor]
    remainder_tokens = tokenized_contents[closest_ctxlen_factor:]

    # We do array_split rather than split here so that `tokens` will have type `list`.
    tokenized_contents = np.array_split(splitable_tkn_chunks, (closest_ctxlen_factor // max_length))

    # ...then append what's left, if it's unevenly divided
    if num_tokens > closest_ctxlen_factor:
        tokenized_contents.append(remainder_tokens)
    
    LOG.info(f"Done! File {filepath} has been tokenized.")

    return tokenized_contents, num_tokens

def _save_as_arrow_file(tokens: list[np.array], output_file: str) -> None:
    '''
    Saves a list of arrays with `context_length` length (unless it is not)

    Params:
    tokens: A list of numpy arrays containing tokens, each with a length of the model's context size.
    output_file: The path of the file which will be saved.
    '''
    LOG.info(f"Writing to arrow file and saving...")
    schema = pa.schema([pa.field('input_ids', type=pa.list_(pa.int64()))])

    with pa.OSFile(output_file, 'wb') as sink:
        with pa.ipc.new_file(sink, schema=schema) as writer:
            for tkn_chunk in tokens:
                batch = pa.RecordBatch.from_pylist([{"input_ids": tkn_chunk}])
                writer.write_batch(batch)

if __name__ == "__main__":
    main()
