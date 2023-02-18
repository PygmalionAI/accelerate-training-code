import os
import logging
import pickle
import struct
import torch
import argparse
import numpy as np
import transformers
import json
import tqdm
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple

logger = logging.getLogger(__name__)

def decode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    mem = np.memmap(in_file, mode="r", dtype="uint16")
    tokens = len(mem)
    with open(out_file, "a") as f:
        for token in tqdm.tqdm(mem):
            f.write(tokenizer.decode([token]))
    return tokens

def encode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> None:
    with open(out_file, "ab") as w:
        with open(in_file, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                w.write(np.uint16(tokenizer.encode(line)))

class TokenizedDataset(torch.utils.data.Dataset):
    """
    Consumes a flat binary file containing 16-bit token serialization, aligned
    along `context_length` chunks.
    """

    def __init__(self, path: str, context_length: int = 2048):
        file_stat = os.stat(path)
        self.file = open(path, 'rb')
        self.length = int(file_stat.st_size / 2 / context_length)
        self.formatstr = '%sH' % context_length
        self.context_length = context_length
        length_mb = os.stat(path).st_size / 1024.0 / 1024.0
        num_tokens = self.length * context_length
        print(f"DATASET: {path}")
        print(f"DATASET SIZE: {length_mb:,.2f}mb, {num_tokens:,} tokens, "
              f"{self.length:,} contexts")

    def __len__(self) -> int:
        return self.length

    def load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.seek(idx)
        input_ids = torch.tensor(
            struct.unpack(self.formatstr,
                          self.file.read(self.context_length * 2)))
        mask = torch.zeros(self.context_length)
        return input_ids, mask

    def seek(self, idx):
        self.file.seek(self.context_length * idx * 2)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.load(idx)

class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, feedback_file: str, tokenizer: transformers.AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feedback_file = feedback_file

        with open(feedback_file) as f:
            self.feedback = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.feedback)

    def __getitem__(self, idx):
        feedback = self.feedback[idx]
        feedback_input = '\n'.join(feedback["input"].split("\n")[-2:])
        feedback_str = f'{feedback_input} {feedback["output"].lstrip().rstrip()}'
        seq = self.tokenizer(
            feedback_str,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
            )
        reward = torch.tensor([feedback["reward"]]).unsqueeze(0)
        return seq, reward

# sft file example
# {
#     "input": "Anonymous: Hi, how are you?\nGPT:",
#     "output": " I'm good, how are you?\n",
#     "reward": 0.0
# }
import tqdm
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, sft_file: str, tokenizer: transformers.AutoTokenizer, is_main_process: bool, max_length: int = 2048, keep_negative_examples: bool = False):
        tokenized_data_cache = f"{sft_file}.tokenized.bin"
        if os.path.isfile(tokenized_data_cache):
            with open(tokenized_data_cache, "rb") as file:
                self.df = pickle.load(file)
            return

        # TODO(11b): nb_workers should default to `num_cores / num_gpus`
        pandarallel.initialize(progress_bar=is_main_process, nb_workers=4, verbose=1)

        self.tokenizer = tokenizer
        self.max_length = max_length

        if is_main_process:
            logger.info("Loading SFT dataset entirely into memory...")

        df = pd.read_json(sft_file, lines=True)

        # Remove negative examples from the dataset if they're not wanted
        if not keep_negative_examples:
            pd = pd[pd["reward"] >= 1.]

        if is_main_process:
            logger.info("Tokenizing SFT dataset...")

        # Length warning messes up progress bars, so we silence temporarily.
        # https://github.com/huggingface/transformers/issues/991
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        df = df.parallel_apply(self._sftrow2item, axis=1)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

        if is_main_process:
            logger.info("Iterating over SFT dataset and dropping long sequences")

        self.df = pd.DataFrame()
        self.df["input_ids"] = df.map(lambda x: x["input_ids"])
        self.df["start_positions"] = df.map(lambda x: x["start_positions"])
        self.df["end_positions"] = df.map(lambda x: x["end_positions"])
        self.df["reward"] = df.map(lambda x: x["reward"])

        # iterate over sft, removing any that have too many tokens
        self.df = self.df.loc[self.df["input_ids"].map(lambda x: len(x[0])) <= self.max_length]

        if is_main_process:
            logger.info("Data is ready")
            with open(tokenized_data_cache, "wb") as file:
                pickle.dump(self.df, file, protocol=pickle.HIGHEST_PROTOCOL)


    def _sftrow2item(self, sft):
        sft_input_tokens = self.tokenizer(sft["input"].strip(), return_tensors="pt").input_ids
        sft_output_tokens = self.tokenizer(f' {sft["output"].strip()}<|endoftext|>', return_tensors="pt").input_ids
        input_ids = torch.cat([sft_input_tokens, sft_output_tokens], dim=-1)
        reward = sft["reward"]

        start_positions = torch.tensor([len(sft_input_tokens[0])])
        end_positions = torch.tensor([len(sft_input_tokens[0]) + len(sft_output_tokens[0]) - 1])
        return {
            "input_ids": input_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "reward": reward,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--in_file', type=str, help='input file to use', required=True)
    parser.add_argument('--out_file', type=str, help='output file to use', required=True)
    parser.add_argument('--model', type=str, help='model tokenizer to use', required=True)
    args = parser.parse_args()

    encode(args.in_file, args.out_file, transformers.AutoTokenizer.from_pretrained(args.model))
