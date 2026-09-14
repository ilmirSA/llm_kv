from transformers import BitsAndBytesConfig
import torch



import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Несколько задач, включая многошаговую математическую — на ней разница заметнее всего
questions = [
    "Какая столица у Франции? Ответь одним словом.",
    "Переведи на английский одним словом: «собака».",
    "Реши по шагам: ноутбук стоил 80 000 рублей. Цену подняли на 15%, затем на "
    "распродаже снизили на 25%, а при оплате картой дали ещё скидку 10%. "
    "Сколько заплатит покупатель картой? В конце назови итоговую сумму в рублях.",
    "Напиши одно предложение о пользе сна.",
]


def load_model(use_int4: bool):
    """Загружает модель в float16 или INT4. Возвращает модель."""
    if use_int4:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",       # NF4 — рекомендованный тип для весов LLM
        bnb_4bit_use_double_quant=True,  # Двойная квантизация даёт ещё немного экономии
        )

        # Ваш код здесь: загрузите модель с quantization_config=bnb_config и device_map="cuda"
        model = AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb_config, device_map="cuda")
    else:
        # Ваш код здесь: загрузите модель в float16 (torch_dtype=torch.float16) и device_map="cuda"
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    return model


def benchmark(model) -> dict:
    """Замеряет память под модель и генерацию ответов на все вопросы."""
    torch.cuda.synchronize()
    mem_gb = torch.cuda.memory_allocated() / 1024**3

    # Прогрев: первый вызов после загрузки платит за инициализацию CUDA-ядер
    # и не должен попадать в замер (см. урок про KV-cache)
    warmup = tokenizer("Привет", return_tensors="pt").to("cuda")
    model.generate(**warmup, max_new_tokens=5, do_sample=False)

    answers = []
    total_new_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for q in questions:
        # Instruct-модель ждёт chat-разметку: без apply_chat_template она
        # работает как «продолжатель текста», а не как ассистент
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]  # Только сгенерированная часть
        total_new_tokens += len(new_tokens)
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "memory_gb": mem_gb,
        "time_sec": elapsed,
        "new_tokens": total_new_tokens,
        "tok_per_sec": total_new_tokens / elapsed,
        "answers": answers,
    }


# float16: загрузили, замерили и сразу выгрузили из памяти
model_fp16 = load_model(use_int4=False)
result_fp16 = benchmark(model_fp16)
del model_fp16            # Освобождаем здесь, в основном коде, иначе модель останется в памяти
gc.collect()
torch.cuda.empty_cache()

model_fp16 = load_model(use_int4=False)
result_fp16 = benchmark(model_fp16)
del model_fp16
gc.collect()
torch.cuda.empty_cache()

# INT4
model_int4 = load_model(use_int4=True)
result_int4 = benchmark(model_int4)
del model_int4
gc.collect()
torch.cuda.empty_cache()

print(f"{'Режим':<10}{'Память, ГБ':<13}{'Время, сек':<12}{'Токенов':<10}{'Ток/с':<8}")
print(f"{'float16':<10}{result_fp16['memory_gb']:<13.2f}{result_fp16['time_sec']:<12.2f}"
      f"{result_fp16['new_tokens']:<10}{result_fp16['tok_per_sec']:<8.1f}")
print(f"{'INT4':<10}{result_int4['memory_gb']:<13.2f}{result_int4['time_sec']:<12.2f}"
      f"{result_int4['new_tokens']:<10}{result_int4['tok_per_sec']:<8.1f}")
print(f"\nINT4 меньше по памяти в {result_fp16['memory_gb'] / result_int4['memory_gb']:.1f}x\n")

for q, a_fp16, a_int4 in zip(questions, result_fp16["answers"], result_int4["answers"]):
    print(f"Вопрос: {q}")
    print(f"  float16: {a_fp16}")
    print(f"  INT4:    {a_int4}")
    print() 