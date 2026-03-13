#!/usr/bin/env python3


import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   


MODEL_NAME = "llama-3.3-70b-versatile"


groq_native = Groq(api_key=GROQ_API_KEY)


groq_via_openai = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)




def query_llama_groq_native(prompt, system_prompt="You are a helpful AI assistant."):
    """Query Llama 3.3 70B using Groq's native Python library."""
    try:
        response = groq_native.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error (Groq native): {e}"

def query_llama_groq_via_openai(prompt, system_prompt="You are a helpful AI assistant."):
    """Query Llama 3.3 70B using OpenAI library pointing to Groq."""
    try:
        response = groq_via_openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error (Groq via OpenAI): {e}"

def compare_responses(prompt):
    """Compare responses from both methods for the same prompt."""
    print(f"Prompt: {prompt}\n")
    print("--- Response from Groq native ---")
    print(query_llama_groq_native(prompt))
    print("\n--- Response from Groq via OpenAI client ---")
    print(query_llama_groq_via_openai(prompt))
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Llama 3.3 70B Integration Demo")
    print("=" * 60 + "\n")

    print("--- Step 1: Problem Definition and Goals ---")
    prompt1 = "Explain how to define a problem and set goals for an AI project. Provide a concise example."
    print(query_llama_groq_native(prompt1))
    print("\n" + "-" * 50 + "\n")

    print("--- Step 2: Choosing the Right AI Approach ---")
    prompt2 = "How do you choose the right AI approach (e.g., deep learning, transfer learning, GNNs) for a given problem? Give an example."
    print(query_llama_groq_via_openai(prompt2))
    print("\n" + "-" * 50 + "\n")

 
    print("--- Step 3: Data Collection and Preprocessing ---")
    prompt3 = "What are key considerations for collecting and preprocessing data for AI? Mention data quality and availability issues."
    print(query_llama_groq_native(prompt3))
    print("\n" + "-" * 50 + "\n")

   
    print("--- Step 4: Knowledge Graph Development ---")
    prompt4 = "Explain how to develop a knowledge graph for an AI system. What role does it play in integrating domain knowledge?"
    print(query_llama_groq_via_openai(prompt4))
    print("\n" + "-" * 50 + "\n")

  
    print("--- Step 5: Model Training ---")
    prompt5 = "Describe the process of training a machine learning model, covering deep learning, transfer learning, graph neural networks, and attention mechanisms."
    print(query_llama_groq_native(prompt5))
    print("\n" + "-" * 50 + "\n")

    print("--- Step 6: Domain Knowledge Integration ---")
    prompt6 = "How can domain-specific knowledge be integrated into an AI model? Provide examples."
    print(query_llama_groq_via_openai(prompt6))
    print("\n" + "-" * 50 + "\n")


    print("--- Step 7: Evaluation, Refinement, and Explainability ---")
    prompt7 = "Explain evaluation and refinement of AI systems. Include the importance of explainability techniques and how to handle ethics and bias."
    print(query_llama_groq_native(prompt7))
    print("\n" + "-" * 50 + "\n")

    print("--- Step 8: Deployment and Maintenance ---")
    prompt8 = "Discuss deployment and maintenance of AI systems. Address challenges like adversarial attacks and ongoing monitoring."
    print(query_llama_groq_via_openai(prompt8))
    print("\n" + "-" * 50 + "\n")


    print("=== Comparison: Both API Methods on the Same Prompt ===")
    compare_prompt = "What are the main challenges in ensuring explainability and transparency in AI?"
    compare_responses(compare_prompt)

    print("\nDemo finished. Remember to check your API usage limits and model availability on Groq.")