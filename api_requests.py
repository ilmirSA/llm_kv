from openai import OpenAI

client = OpenAI(base_url="http://158.160.11.102:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Кратко объясни, что такое контейнеризация."}],
    max_tokens=80,
    temperature=0,
)
print(resp.choices[0].message.content)