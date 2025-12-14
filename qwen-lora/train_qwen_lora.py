import modal

app = modal.App("qwen_lora_trainer")

# Use existing volume
volume = modal.Volume.from_name("qwen-data", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "peft",
        "trl",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={"/data": volume},   # volume mounted at /data
)
def train():
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer
    from peft import LoraConfig

    # --- 1. Load dataset ---
    DATASET_PATH = "/data/root/data/lora_messages_dataset.jsonl"
    print("Loading dataset from:", DATASET_PATH)

    ds = load_dataset("json", data_files=DATASET_PATH, split="train")

    # --- 2. Load base model (no 4-bit, no bitsandbytes) ---
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"   # smaller than 3B, easier to train

    print("Loading model:", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare LoRA config ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # --- 4. Convert your `messages` → plain text using chat template ---
    def format_example(example):
        # example["messages"] is your list of {role, content}
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("Formatting dataset with chat template...")
    ds_formatted = ds.map(format_example)

    # --- 5. Trainer ---
    training_args = TrainingArguments(
        output_dir="/data/root/output-model",   # save into volume so you can fetch it
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        max_steps=600,          # tweak later if needed
        fp16=True,
        optim="adamw_torch",
    )

    print("Starting LoRA training...")
    trainer = SFTTrainer(
    model=model,
    train_dataset=ds_formatted,   # already converted text dataset
    peft_config=peft_config,
    args=training_args,
    )




    trainer.train()
    print("Training finished, saving model...")
    trainer.save_model("/data/root/output-model")
    print("✅ LoRA model saved to /data/root/output-model (inside qwen-data volume)")

