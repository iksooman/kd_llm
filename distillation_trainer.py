import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import Accelerator

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Placeholder for Llama 3.2 1B. Replace with actual Llama 3.2 1B when available.
DATA_FILE = "synthetic_qa_data.jsonl"
OUTPUT_DIR = "./distilled_model"
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4 # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 1 # Adjust based on GPU memory
MAX_SEQ_LENGTH = 512 # Max sequence length for tokenization

def load_data(file_path):
    """Loads data from a JSONL file and converts it to a Hugging Face Dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def prepare_model_and_tokenizer(model_name):
    """Loads the student model and tokenizer, and prepares for PEFT."""
    accelerator = Accelerator()

    # Load model with 4-bit quantization if CUDA is available
    bnb_config = None
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        print("Loading model with 4-bit quantization...")
    else:
        print("CUDA not available, loading model in full precision.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index} if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Llama models prefer right padding

    # Prepare model for k-bit training
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

def formatting_prompts_func(examples):
    """Formats the input for the student model.
    The SFTTrainer will handle the loss calculation on the 'answer' part.
    """
    texts = []
    for i in range(len(examples["question"])):
        text = f"### Question:\n{examples['question'][i]}\n\n### Answer:\n{examples['answer'][i]}"
        texts.append(text)
    return {"text": texts}

def train_distilled_model():
    """Main function to perform knowledge distillation."""
    print(f"Loading data from {DATA_FILE}...")
    dataset = load_data(DATA_FILE)
    print(f"Loaded {len(dataset)} examples.")

    print(f"Preparing model and tokenizer: {MODEL_NAME}...")
    model, tokenizer = prepare_model_and_tokenizer(MODEL_NAME)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_steps=100,
        fp16=True if torch.cuda.is_available() else False, # Use fp16 if CUDA is available
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch", # Use paged_adamw_8bit for 4-bit training
        report_to="wandb", # Report to Weights & Biases
        push_to_hub=False,
        remove_unused_columns=False, # Keep columns for formatting_prompts_func
    )

    # SFTTrainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config, # Pass the peft_config from the model
        dataset_text_field="text", # This will be populated by formatting_prompts_func
        max_seq_length=MAX_SEQ_LENGTH, # Max sequence length for tokenization
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Distilled model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_distilled_model()
