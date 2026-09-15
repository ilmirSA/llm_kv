import time
import gc
import torch
from transformers import pipeline, GenerationConfig

MODEL = "Qwen/Qwen2.5-3B-Instruct"

pipe = pipeline("text-generation", model=MODEL, torch_dtype=torch.float16, device_map="cuda")

# Чистый конфиг генерации: задаём только max_new_tokens
# Иначе pipeline подтянет max_length из конфига модели и будет сыпать предупреждениями
gen_config = GenerationConfig(max_new_tokens=80, do_sample=False)

prompt = "Кратко объясни, что такое контейнеризация."

# Instruct-модель ждёт chat-разметку — собираем вход так же, как это сделает
# vLLM на /v1/chat/completions: обе стороны бенчмарка получают одну и ту же нагрузку
chat_text = pipe.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)

# Прогрев: первый вызов платит за инициализацию CUDA-ядер, в замер он входить не должен
_ = pipe(chat_text, generation_config=GenerationConfig(max_new_tokens=5, do_sample=False))


def benchmark_pipeline(total: int = 16) -> dict:
    """transformers.pipeline обрабатывает запросы последовательно, один за другим.
    Прогоняем total запросов и считаем суммарное время, throughput
    и среднее время обслуживания одного запроса. Готовая функция замера - менять не нужно."""
    latencies = []
    start = time.perf_counter()
    for _ in range(total):
        t0 = time.perf_counter()
        pipe(chat_text, generation_config=gen_config)
        latencies.append(time.perf_counter() - t0)  # время одного запроса
    elapsed = time.perf_counter() - start
    return {
        "total": total,
        "time_sec": elapsed,
        "throughput": total / elapsed,
        "avg_service": sum(latencies) / len(latencies),
    }


pipeline_result = benchmark_pipeline(total=16)
print(f"pipeline | {pipeline_result['total']} запросов подряд | "
      f"{pipeline_result['time_sec']:.1f} с | "
      f"{pipeline_result['throughput']:.2f} запр/с | "
      f"обслуживание {pipeline_result['avg_service']:.2f} с/запрос")

# Выгружаем пайплайн, освобождаем память под vLLM
del pipe
gc.collect()
torch.cuda.empty_cache()
print(f"Память после выгрузки: {torch.cuda.memory_allocated() / 1024**3:.2f} ГБ")