import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://158.160.11.102:8000/v1", api_key="EMPTY")

prompt = "Кратко объясни, что такое контейнеризация."


async def vllm_request(semaphore):
    submitted = time.perf_counter()      # Запрос «отправлен пользователем»
    async with semaphore:
        start = time.perf_counter()      # Сервер взял запрос в работу
        resp = await client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0,
        )
        service = time.perf_counter() - start           # Время обслуживания
        user_latency = time.perf_counter() - submitted  # Ожидание пользователя
        return user_latency, service, resp.choices[0].message.content


async def load_test_vllm(concurrency: int, total: int = 16) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    tasks = [vllm_request(semaphore) for _ in range(total)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    user_latencies = [u for u, _, _ in results]
    service_times = [s for _, s, _ in results]
    return {
        "concurrency": concurrency,
        "time_sec": elapsed,
        "throughput": total / elapsed,
        "avg_user_latency": sum(user_latencies) / len(user_latencies),
        "avg_service": sum(service_times) / len(service_times),
    }


async def main():
    for c in [1, 4, 16]:
        r = await load_test_vllm(c)
        print(f"vLLM | concurrency={r['concurrency']:>2} | "
            f"{r['time_sec']:6.1f} с | {r['throughput']:.2f} запр/с | "
            f"ожидание {r['avg_user_latency']:5.2f} с | "
            f"обслуживание {r['avg_service']:.2f} с")

if __name__ == '__main__':
    asyncio.run(main())