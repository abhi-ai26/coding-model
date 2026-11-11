"""Week 1, Day 4: LoRA fine-tune Qwen2.5-Coder-1.5B on 5k OpenCoder samples with Unsloth."""
import os
os.environ.setdefault("WANDB_PROJECT", "week1")
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tok = FastLanguageModel.from_pretrained("/workspace/hf/hub/models--unsloth--Qwen3-30B-A3B-Base-bnb-4bit/snapshots/f26f24d1a7f007c3bf3e8530a665d66d7b75aace", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train").select(range(5000))
print("columns:", ds.column_names)
def fmt(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}{tok.eos_token}"}
ds = ds.map(fmt, remove_columns=ds.column_names)
trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=ds,
    args=SFTConfig(output_dir="out-sft-30b", per_device_train_batch_size=4, gradient_accumulation_steps=4,
                   num_train_epochs=1, learning_rate=2e-4, logging_steps=10, bf16=True,
                   max_length=2048, report_to="wandb", run_name="w4-qlora-30b-a3b"),
)
trainer.train()
model.save_pretrained("qwen3-30b-a3b-qlora-adapter"); tok.save_pretrained("qwen3-30b-a3b-qlora-adapter")
print("saved adapter to qwen3-30b-a3b-qlora-adapter")
