import tiktoken

def audit_prompts(prompts, model="gpt-4o", input_rate_per_1m=5.00, output_rate_per_1m=15.00):
    enc = tiktoken.encoding_for_model(model)
    audit_data = []
    
    
    
    
    
    for i, prompt in enumerate(prompts, 1):
        input_tokens = len(enc.encode(prompt))
        # Estimate average response tokens (assume ~20% of input or a fixed 300 tokens)
        output_tokens = 300 
        
        est_cost = (input_tokens / 1e6 * input_rate_per_1m) + (output_tokens / 1e6 * output_rate_per_1m)
        audit_data.append({
            "Prompt ID": f"P{i}",
            "Input Tokens": input_tokens,
            "Est. Output Tokens": output_tokens,
            "Est. Cost ($)": round(est_cost, 6)
        })
    return audit_data

# Example prompts for testing
sample_prompts = [
    "Summarize this short paragraph: " + "AI is changing the world. " * 10,
    "RAG query with context: " + "Document snippet " * 200,
    "Complex multi-step reasoning: " + "Analyze financial statement line item... " * 100,
]

print(audit_prompts(sample_prompts))
