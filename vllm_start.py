import subprocess
import time
import requests

# Вывод сервера направляем в файл, чтобы при проблемах видеть, что пошло не так
log_file = open("vllm_server.log", "w")
server = subprocess.Popen(
    ["vllm", "serve", "Qwen/Qwen2.5-3B-Instruct",
     "--max-model-len", "2048",
     "--gpu-memory-utilization", "0.85",
     "--dtype", "half",
     "--enforce-eager",
     "--port", "8000"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)


def wait_until_ready(url: str, timeout: int = 600) -> bool:
    """Опрашивает сервер, пока он не ответит, что готов принимать запросы.
    Если процесс упал, сразу сообщает об этом, не дожидаясь таймаута."""
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            print(f"Сервер завершился с кодом {server.poll()}. Смотрите vllm_server.log")
            return False
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


# /v1/models отвечает 200, когда модель загружена и сервер готов
if wait_until_ready("http://localhost:8000/v1/models"):
    print("vLLM-сервер готов принимать запросы")
else:
    print("Сервер не поднялся - смотрите vllm_server.log")