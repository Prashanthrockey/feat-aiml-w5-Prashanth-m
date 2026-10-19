
import json
from pathlib import Path

import requests

root = Path(__file__).resolve().parent.parent
output_folder = root / "outputs"
output_folder.mkdir(exist_ok=True)

url = "http://localhost:11434/api/generate"

system_prompt = (
    "You are a helpful AI and machine learning tutor. "
    "Give clear and beginner-friendly answers."
)


def ask_model(model, prompt):
    response = requests.post(
        url,
        json={
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]


prompts = [
    "What is machine learning?",
    "Explain Python lists in simple words.",
    "What is an API?",
    "Explain embeddings in two sentences.",
    "What is the difference between AI and ML?"
]

prompt_results = []

print("FIVE PROMPT TESTS")

for prompt in prompts:
    answer = ask_model("llama3.2:3b", prompt)

    print("\nPrompt:", prompt)
    print("Answer:", answer)

    prompt_results.append({
        "model": "llama3.2:3b",
        "prompt": prompt,
        "response": answer
    })

with open(output_folder / "prompt_results.json", "w", encoding="utf-8") as file:
    json.dump(prompt_results, file, indent=2, ensure_ascii=False)

questions = [
    "Explain machine learning to a beginner.",
    "What is an embedding, and why is it useful?",
    "Explain how a vector database helps a Q&A bot."
]

models = ["llama3.2:3b", "qwen2.5:3b"]
comparison_results = []

print("\n\nMODEL COMPARISON")

for question in questions:
    print("\nQuestion:", question)

    for model in models:
        answer = ask_model(model, question)

        print("\nModel:", model)
        print("Answer:", answer)

        comparison_results.append({
            "question": question,
            "model": model,
            "response": answer
        })

with open(output_folder / "model_comparison.json", "w", encoding="utf-8") as file:
    json.dump(comparison_results, file, indent=2, ensure_ascii=False)

print("\nBoth tests completed successfully.")
print("Results saved in the outputs folder.")
