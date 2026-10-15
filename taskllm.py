from transformers import pipeline


pipe = pipeline(
    "text-generation",
    model="./models/qwen",
    device=-1,
    clean_up_tokenization_spaces=False,
)

questions = [
    "Привет! Как дела?",
    "Объясни в двух предложениях, что такое машинное обучение.",
    "Назови три столицы европейских стран.",
]

for q in questions:
    print(f" {q}")
    result = pipe(
        q,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )
    answer = result[0]["generated_text"].strip()
    print(f" {answer}\n")
    print("-" * 60)