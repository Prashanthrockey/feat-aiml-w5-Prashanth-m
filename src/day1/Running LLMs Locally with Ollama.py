
import json
from pathlib import Path

import requests

API_URL = "http://localhost:11434/api/generate"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

SYSTEM_PROMPT = """
You are an AI and machine learning tutor.
Answer accurately using simple English.
Follow the instructions in each question.
"""

QUESTIONS = [
    "Explain machine learning in three sentences.",
    "Write a Python function to check whether a number is even.",
    "Give two advantages and two limitations of running an LLM locally."
]


def generate_answer(model, question):
    response = requests.post(
        API_URL,
        json={
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": question,
            "stream": False,
            "options": {"temperature": 0.3}
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]


def main():
    results = []
    failures = 0

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {index}: {question}")

        for model in MODELS:
            try:
                answer = generate_answer(model, question)

                print(f"\nModel: {model}")
                print(answer)

                results.append({
                    "question_number": index,
                    "question": question,
                    "model": model,
                    "response": answer
                })

            except requests.RequestException as error:
                failures += 1
                print(f"Error with {model}: {error}")

    output_file = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "model_comparison.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSuccessful responses: {len(results)}")
    print(f"Failed requests: {failures}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
