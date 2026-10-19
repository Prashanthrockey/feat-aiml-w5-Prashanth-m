
import requests

url = "http://localhost:11434/api/generate"

system_prompt = "You are a helpful AI tutor. Answer briefly and clearly."

prompts = [
    "Explain prompt engineering.",
    "What is Ollama?",
    "Explain machine learning.",
    "What is a Python list?",
    "Why run an LLM locally?"
]

for i, prompt in enumerate(prompts, 1):
    print(f"\nTest {i}: {prompt}")

    try:
        response = requests.post(
            url,
            json={
                "model": "llama3.2:3b",
                "system": system_prompt,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        print(response.json()["response"])

    except requests.RequestException as error:
        print(f"Error: {error}")

print("\nEvaluation finished.")
