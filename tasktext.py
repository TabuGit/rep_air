from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="./models/albert-base-v2",
    tokenizer="./models/albert-base-v2",
)

texts = [
    "I love this pizza!",
    "This movie is terrible.",
    "The weather is okay today.",
]

for t in texts:
    r = sentiment(t)[0]
    print(f"{t:35s} -> {r['label']} ({r['score']:.3f})")

generator = pipeline(
    "text-generation",
    model="./models/gpt2",
    tokenizer="./models/gpt2",
    clean_up_tokenization_spaces=False,
)

prompt = "The future of artificial intelligence is"
output = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1,
    pad_token_id=generator.tokenizer.eos_token_id,
    return_full_text=False,
)
print(output[0]["generated_text"])