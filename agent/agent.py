"""
=============================================================
  GreenEnergy — Smart EV Assistant
  Agent Core — loads fine-tuned model and answers questions
=============================================================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
import os

# ── Configuration ──────────────────────────────────────────
BASE_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_MODEL = "mamoon94/greenenergy-ev-assistant"

SYSTEM_PROMPT = """You are GreenEnergy, a Smart EV Assistant with expert knowledge of \
electric vehicle technology, battery systems, charging infrastructure, grid optimization, \
vehicle specifications, performance metrics, costs, and sustainability. \
Provide accurate, detailed, and helpful answers about all aspects of electric vehicles. \
Always support your answers with specific data and metrics where available."""

# ── Load model ─────────────────────────────────────────────
print("Loading GreenEnergy model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Check if GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
else:
    # CPU mode for testing
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)

model.eval()
print("✅ GreenEnergy model loaded!")


# ── Conversation memory ────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.history = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Keep only last N turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []


# ── Main agent function ────────────────────────────────────
def ask_greenenergy(
    question: str,
    memory: Optional[ConversationMemory] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Ask GreenEnergy a question about electric vehicles.

    Args:
        question: The user's question
        memory: Optional conversation memory for multi-turn chat
        max_new_tokens: Maximum tokens to generate
        temperature: Controls randomness (0.1=focused, 1.0=creative)

    Returns:
        The assistant's answer as a string
    """

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history if memory exists
    if memory:
        messages.extend(memory.get_history())

    # Add current question
    messages.append({"role": "user", "content": question})

    # Format using LLaMA 3.1 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Save to memory if provided
    if memory:
        memory.add("user", question)
        memory.add("assistant", response)

    return response


# ── Test the agent ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  GreenEnergy — Smart EV Assistant")
    print("="*50)

    memory = ConversationMemory()

    test_questions = [
        "What is the range of Tesla Model 3?",
        "How does V2G technology work?",
        "Which EV has the best battery in 2025?",
    ]

    for q in test_questions:
        print(f"\n❓ {q}")
        answer = ask_greenenergy(q, memory)
        print(f"🤖 {answer}\n")
        print("-" * 50)
