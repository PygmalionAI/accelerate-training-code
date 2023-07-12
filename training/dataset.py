import pyarrow as pa
import torch

from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

# NOTE(11b): Needs to be kept in sync with the data tokenization script.
IGNORE_INDEX = -100

class MmappedArrowDataset(Dataset):
    '''Memory-mapped PyArrow dataset.'''
    def __init__(self, filepath: str, sft: bool = True) -> None:
        source = pa.memory_map(filepath, "r")
        self.reader = pa.ipc.RecordBatchFileReader(source)
        self.sft = sft

    def __len__(self) -> int:
        return self.reader.num_record_batches

    def __getitem__(self, idx) -> dict:
        # TODO(TG): This is an abomination.
        if self.sft:
            return dict(
                input_ids=self.reader.get_batch(idx)['input_ids'][0],
                labels=self.reader.get_batch(idx)['labels'][0]
            )
        else:
            return dict(
                input_ids=self.reader.get_batch(idx)['input_ids'][0]
            )

class DataCollatorForMmapedDataset():
    def __init__(self, tokenizer: PreTrainedTokenizer, sft: bool = True) -> None:
        self.tokenizer = tokenizer
        self.sft = sft
        self.pad_token_id: int = self.tokenizer.pad_token_id \
            if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id # type: ignore

    def __call__(self, instances) -> dict:
        if self.sft:
            input_ids = [
                torch.tensor(instance["input_ids"].as_py())
                for instance in instances
            ]
        else:
            # Int64Array has a different method to convert itself
            # to a native Python list.
            input_ids = [
                torch.tensor(instance["input_ids"].to_pylist())
                for instance in instances
            ]
        
        # NOTE(TG): Tensor cores are most efficient when dealing with tensor lengths that are multiples of 8.
        # Therefore, we add a fake tensor to the batches so that rnn.pad_sequence
        # will pad to that length.
        input_ids_pad_tensor = self._create_fake_padding_tensor(input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [*input_ids, input_ids_pad_tensor], batch_first=True, padding_value=self.pad_token_id)
        # Remove fake tensor now that its purpose is served
        # (praying the fake tensor is always the last in the batch size)
        input_ids = input_ids[:-1]
        
        if self.sft:
            labels = [
                torch.tensor(instance["labels"].as_py()) for instance in instances
            ]
            labels_pad_tensor = self._create_fake_padding_tensor(labels)
            labels = torch.nn.utils.rnn.pad_sequence(
                [*labels, labels_pad_tensor], batch_first=True, padding_value=IGNORE_INDEX)
            labels = labels[:-1]
        else:
            # In UFT, labels are the same as the input_ids
            labels = input_ids
        
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
    
    def _create_fake_padding_tensor(self, sequences: torch.Tensor) -> torch.Tensor:
        '''Makes a fake 'padding tensor' that has a length of a multiple of 8 to a sequence of tensors.'''
        # https://stackoverflow.com/questions/72540912/find-the-biggest-of-two-pytorch-tensor-on-size
        longest_seq_len = max(sequences, key=len).shape[0]

        # Find closest multiple of 8 to the longest seq_len
        if longest_seq_len % 8 != 0:
            next_multiple = (longest_seq_len // 8 + 1) * 8
        else:
            # Make a fake tensor anyway so that we don't have to check
            # in the main __call__ function to see if a fake tensor was added or not
            next_multiple = longest_seq_len

        fake_tensor = torch.ones((next_multiple), dtype=sequences[0].dtype)
        return fake_tensor
