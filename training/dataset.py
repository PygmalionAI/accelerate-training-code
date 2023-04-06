import pyarrow as pa
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

# NOTE(11b): Needs to be kept in sync with the data tokenization script.
IGNORE_INDEX = -100


class MmappedArrowDataset(Dataset):

    def __init__(self, filepath: str):
        source = pa.memory_map(filepath, "r")
        reader = pa.ipc.RecordBatchFileReader(source)
        self.table = reader.read_all()

    def __len__(self):
        return len(self.table)
        return self.reader.num_record_batches

    def __getitem__(self, idx):
        return dict(
            input_ids=self.table["input_ids"][idx],
            labels=self.table["labels"][idx],
        )
        return self.reader.get_batch(idx)


class DataCollatorForMmapedDataset():

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id: int = self.tokenizer.pad_token_id \
            if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id # type: ignore

    def __call__(self, instances):
        input_ids = [
            torch.tensor(instance["input_ids"].as_py())
            for instance in instances
        ]

        labels = [
            torch.tensor(instance["labels"].as_py()) for instance in instances
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,

            # NOTE(11b): First approach will likely go wrong because EOS will
            # be the padding token but we _do_ want to train on it (so the model
            # learns when to stop generating). Second approach should work, but
            # we don't really need it since `labels` will already have
            # IGNORE_INDEX for the proper positions.

            # attention_mask=input_ids.ne(self.pad_token_id),
            # attention_mask=labels.ne(IGNORE_INDEX),
        )
