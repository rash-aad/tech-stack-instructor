import modal

app = modal.App("qwen_lora_inference")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers==4.44.0",
        "accelerate",
        "torch",
        "peft==0.16.0",
        "bitsandbytes"
    )
)

@app.function(
    image=image,
    gpu="a10g",
    volumes={"/model": modal.Volume.from_name("qwen-data")},
    timeout=600,
)
def infer(prompt: str):

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    print("Loading base model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        "/model/root/output-model"
    )
    model = model.to("cuda")

   
    clean_prompt = f"""
Return output ONLY in this exact format:

role: <role>
experience: <years>
keywords: <comma-separated list>

NO extra text.
NO code.
NO explanations.

INPUT: {prompt}
"""

    print("Generating output...")
    encoded = tokenizer(clean_prompt, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        **encoded,
        max_new_tokens=120,
        temperature=0.01,   
        top_p=0.9
    )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n===== RESULT =====\n")
    print(result)
    print("\n==================\n")

