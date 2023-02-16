import copy
import os
import torch
import accelerate
import tqdm
import time
import argparse
import json

from dataset import TokenizedDataset, FeedbackDataset, SFTDataset
from lion import Lion

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from typing import Union, Optional

# Have optimizers in a dictionary rather than in an if statement
# This is so that if more optimizers are added in the future,
# the code looks cleaner
OPTIMIZER_DICT = {
    "adamw": torch.optim.AdamW,
    "lion": Lion
}

# Supervised Finetuning: Compute loss between model output and target using start_positions and end_positions
def sft_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    start_positions: Optional[torch.LongTensor] = None,
    end_positions: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[torch.Tensor, CausalLMOutput]:
    try:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    except AttributeError:
        return_dict = True

    outputs = self.module.gpt_neox(
        input_ids,
        attention_mask=attention_mask,

        # TODO(11b): Not used in NeoX. Ideally this code shouldn't be
        # model-specific at all but let's not spend time on that right now.
        #
        # token_type_ids=token_type_ids,
        # position_ids=position_ids,

        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    logits = self.module.embed_out(sequence_output)

    answer_logits = logits[:, start_positions[0]:end_positions[0]+1]
    answer_input_ids = input_ids[:, start_positions[0]:end_positions[0]+1]

    # prompt_logits = logits[:, :start_positions[0]]
    # prompt_input_ids = input_ids[:, :start_positions[0]]

    # compute loss for prompt and answer
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    shift_answer_logits = answer_logits[..., :-1, :].contiguous()
    shift_answer_labels = answer_input_ids[..., 1:].contiguous()
    # shift_prompt_logits = prompt_logits[..., :-1, :].contiguous()
    # shift_prompt_labels = prompt_input_ids[..., 1:].contiguous()
    answer_loss = loss_fct(shift_answer_logits.view(-1, answer_logits.size(-1)), shift_answer_labels.view(-1))
    # prompt_loss = loss_fct(shift_prompt_logits.view(-1, prompt_logits.size(-1)), shift_prompt_labels.view(-1))

    # loss = (prompt_loss + answer_loss) / 2
    loss = answer_loss

    if not return_dict:
        output = (loss,) + outputs[2:]
        return ((loss,) + outputs[2:]) if return_dict else output

    return CausalLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class SFT_Trainer:
    def __init__(
        self,
        accelerator: accelerate.Accelerator,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        weight_dtype: torch.dtype,
        args: argparse.Namespace,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.weight_dtype = weight_dtype
        self.args = args
        self.starting_step = 0
        self.local_step = 0

        if args.resume_from is not None:
            self.load_model(args.resume_from)

        if accelerator.is_main_process:
            self.progress_bar = tqdm.tqdm(
                total=self.args.epochs*len(train_dataloader),
                desc="Total Steps",
                leave=False,
            )

    def load_model(self, path) -> None:
        # TODO(11b): Doesn't work with slim checkpoints.
        self.accelerator.load_state(path)

        state_file_path = os.path.join(path, "trainer_state.json")
        with open(state_file_path, "r") as state_file:
            trainer_state = json.load(state_file)
            self.starting_step = trainer_state["step"]

    def save_model(self) -> None:
        path = os.path.join(self.args.output_dir, self.args.run_name)
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)

            state_file_path = os.path.join(path, "trainer_state.json")
            with open(state_file_path, "w") as state_file:
                state_file.write(json.dumps({
                    "step": self.local_step
                }))
        self.accelerator.wait_for_everyone()

        if self.args.save_slim_weights:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save
            )
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(path)
        else:
            self.accelerator.save_state(path)

    def step(self, batch: dict) -> None:
        with self.accelerator.accumulate(self.model):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            start_positions = batch['start_positions'].to("cuda")
            end_positions = batch['end_positions'].to("cuda")

            try:
                outputs = sft_forward(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise e

                print(f"RuntimeError: {e}")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print(f"start_positions: {start_positions}")
                print(f"end_positions: {end_positions}")
                print('Skipping batch...')
                loss = torch.tensor(float('nan'), device=self.accelerator.device)

        return {
            "train/loss": loss.detach().item(),
        }

    def eval_step(self, batch: dict) -> None:
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        start_positions = batch['start_positions'].to("cuda")
        end_positions = batch['end_positions'].to("cuda")

        with torch.no_grad():
            try:
                outputs = sft_forward(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise e

                print(f"RuntimeError: {e}")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print(f"start_positions: {start_positions}")
                print(f"end_positions: {end_positions}")
                print('Skipping batch...')
                loss = torch.tensor(float('nan'), device=self.accelerator.device)

        return loss

    def do_eval(self) -> None:
        if self.eval_dataloader is None:
            return

        self.model.eval()
        eval_losses = []

        if self.accelerator.is_main_process:
            progress_bar = tqdm.tqdm(
                total=len(self.eval_dataloader),
                desc="Eval Steps",
                leave=False,
            )

        for batch in self.eval_dataloader:
            loss = self.eval_step(batch).unsqueeze(0)
            eval_losses.append(loss)
            if self.accelerator.is_main_process:
                progress_bar.update(1)

        eval_losses = torch.cat(eval_losses)
        eval_losses = eval_losses[:len(self.eval_dataloader)]
        eval_loss = torch.mean(eval_losses)
        if self.accelerator.is_main_process:
            # FIXME(11b): This isn't showing up on Tensorboard.
            self.accelerator.log({"eval/loss": eval_loss}, step=self.local_step)

            # Writing to a file so we have this data until the above is fixed.
            path = os.path.join(self.args.output_dir, self.args.run_name)
            with open(os.path.join(path, "eval_losses.csv"), "a") as file:
                file.write(f"{self.local_step};{eval_loss}\n")

        self.model.train()

    def train(self) -> None:
        # Delete some keys from the CLI args because they're prone to info leakage.
        hps = copy.deepcopy(vars(self.args))
        del hps['model']
        del hps['train_dataset']
        del hps['eval_dataset']

        self.accelerator.init_trackers(self.args.run_name, config=hps)
        self.model.train()
        for epoch in range(self.args.epochs):
            for idx, batch in enumerate(self.train_dataloader):
                # Skip over data if resuming a training run.
                if idx < self.starting_step:
                    self.local_step += 1
                    if self.accelerator.is_main_process:
                        self.progress_bar.update(1)
                    continue

                step_start = time.perf_counter()

                #print(f"####\n{self.tokenizer.decode(batch['input_ids'][0])}\n#{batch['start_positions'][0]}:{batch['end_positions'][0]}\n####")

                metrics = self.step(batch)

                step_end = time.perf_counter()
                self.local_step += 1

                if self.accelerator.is_main_process:
                    rank_samples_per_second = self.args.batch_size / (step_end - step_start)
                    world_samples_per_second = rank_samples_per_second * self.accelerator.num_processes

                    metrics.update({
                        "perf/rank_samples_per_second": rank_samples_per_second,
                        "perf/world_samples_per_second": world_samples_per_second,
                        "train/epoch": epoch,
                        "train/samples_seen": self.local_step * self.args.batch_size,
                    })

                    self.progress_bar.update(1)
                    self.progress_bar.set_postfix(**metrics)

                    self.accelerator.log(metrics, step=self.local_step)

                if self.local_step % self.args.save_steps == 0:
                    self.save_model()
                    self.do_eval()
        self.save_model()
        self.do_eval()
        self.accelerator.end_training()

def main() -> None:

    parser = argparse.ArgumentParser(description="Supervised GPT finetuning")
    parser.add_argument("--model", type=str, default="hakurei/gpt-j-random-tinier", help="Model name")
    parser.add_argument("--train_dataset", type=str, default="train.jsonl", help="Training file")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Eval split file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every x steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_slim_weights", action="store_true", help="Save only slim weights when saving checkpoints")
    parser.add_argument("--log_with", type=str, default="all", help="Which experiment tracker to use")
    parser.add_argument("--run_name", type=str, required=True, help="Name of this run, will be used as a folder name")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--resume_from", type=str, help="Resume training from a checkpoint")
    parser.add_argument("--save_pretrained", type=str, help="Save pretrained checkpoint after continuing a training run")
    parser.add_argument("--optimizer", type=str, default="adamw", help="The optimizer to use during model training")
    args = parser.parse_args()

    assert args.optimizer.lower() in OPTIMIZER_DICT.keys(), "Invalid optimizer type specified!"
    
    project_dir = os.path.join(args.output_dir, "logs")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,
        project_dir=project_dir,
    )
    accelerate.utils.set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batches):
        input_ids = [
            batch["input_ids"].squeeze(0) for batch in batches
        ]
        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        start_positions = torch.stack(
            [batch["start_positions"] for batch in batches]
        )
        end_positions = torch.stack(
            [batch["end_positions"] for batch in batches]
        )
        return {
            "input_ids": padded_tokens["input_ids"],
            "attention_mask": padded_tokens["attention_mask"],
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

    train_dataset = SFTDataset(
        args.train_dataset, tokenizer, is_main_process=accelerator.is_main_process)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    eval_dataloader = None
    if args.eval_dataset is not None:
        eval_dataset = SFTDataset(
            args.eval_dataset, tokenizer, is_main_process=accelerator.is_main_process)

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    
    optimizer_cls = OPTIMIZER_DICT[args.optimizer.lower()]

    optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    if args.save_pretrained:
        accelerator.load_state(args.resume_from)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.half()
        unwrapped_model.save_pretrained(
            args.save_pretrained,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.save_pretrained)
        quit()

    trainer = SFT_Trainer(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        weight_dtype=None,
        args=args,
    )

    trainer.train()

if __name__ == '__main__':
    """
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

    # Add supervised finetuning forward method to model
    model.forward = sft_forward.__get__(model)

    # Create input tensors
    question = 'What is the capital of France?'
    answer = 'The capital of France is Paris.'
    question_tokens = tokenizer.encode(question, return_tensors='pt')
    answer_tokens = tokenizer.encode(answer, return_tensors='pt')
    input_ids = torch.cat([question_tokens, answer_tokens], dim=-1)

    start_positions = torch.tensor([len(question_tokens[0])])
    end_positions = torch.tensor([len(question_tokens[0]) + len(answer_tokens[0]) - 1])

    # Compute loss
    loss = model(input_ids, start_positions=start_positions, end_positions=end_positions).loss
    print(loss)
    """
    main()
