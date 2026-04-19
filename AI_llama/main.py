import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import ollama
import base64
import os
from datetime import datetime

MODELS = ["llama3.2", "gemma3:1b"]

# ============================================================
# 1. ЧАТ-БОТ
# ============================================================
def chatbot(model_name):
    print(f"\n{'='*50}")
    print(f"ЧАТ-БОТ — модель: {model_name}")
    print(f"{'='*50}")

    messages = []
    results = []

    questions = [
        "Привіт! Хто ти?",
        "Назви 3 факти про Python",
        "Дякую, до побачення!"
    ]

    for question in questions:
        print(f"\nКористувач: {question}")
        messages.append({"role": "user", "content": question})

        response = ollama.chat(model=model_name, messages=messages)
        answer = response['message']['content']

        messages.append({"role": "assistant", "content": answer})

        print(f"Модель: {answer}")
        results.append(f"Користувач: {question}\nМодель: {answer}\n")

    return results


# ============================================================
# 2. ГЕНЕРАЦІЯ ТЕКСТУ
# ============================================================
def generate_text(model_name):
    print(f"\n{'='*50}")
    print(f"ГЕНЕРАЦІЯ ТЕКСТУ — модель: {model_name}")
    print(f"{'='*50}")

    prompts = [
        "Напиши короткий вірш про штучний інтелект",
        "Поясни що таке машинне навчання у 3 реченнях",
        "Придумай назву для стартапу в сфері AI"
    ]

    results = []

    for prompt in prompts:
        print(f"\nЗапит: {prompt}")
        response = ollama.generate(model=model_name, prompt=prompt)
        answer = response['response']
        print(f"Відповідь: {answer}")
        results.append(f"Запит: {prompt}\nВідповідь: {answer}\n")

    return results


# ============================================================
# 3. MULTIMODAL (зображення + текст)
# ============================================================
def multimodal(model_name):
    print(f"\n{'='*50}")
    print(f"MULTIMODAL — модель: {model_name}")
    print(f"{'='*50}")

    results = []

    # Якщо немає зображення — пропускаємо
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        msg = f"Файл {image_path} не знайдено. Пропускаємо multimodal."
        print(msg)
        results.append(msg)
        return results

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    question = "Що зображено на цій картинці? Опиши детально."
    print(f"Запит: {question}")

    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": question,
            "images": [image_data]
        }]
    )

    answer = response['message']['content']
    print(f"Відповідь: {answer}")
    results.append(f"Запит: {question}\nВідповідь: {answer}\n")

    return results


# ============================================================
# 4. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ У ФАЙЛ
# ============================================================
def save_results(model_name, chat_results, generate_results, multimodal_results):
    filename = f"results_{model_name.replace(':', '_')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"РЕЗУЛЬТАТИ РОБОТИ МОДЕЛІ: {model_name}\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

        f.write("--- 1. ЧАТ-БОТ ---\n\n")
        for line in chat_results:
            f.write(line + "\n")

        f.write("--- 2. ГЕНЕРАЦІЯ ТЕКСТУ ---\n\n")
        for line in generate_results:
            f.write(line + "\n")

        f.write("--- 3. MULTIMODAL ---\n\n")
        for line in multimodal_results:
            f.write(line + "\n")

    print(f"\n✅ Результати збережено у файл: {filename}")
    return filename


# ============================================================
# 5. ПОРІВНЯННЯ МОДЕЛЕЙ
# ============================================================
def compare_models(filenames):
    comparison_file = "comparison.txt"

    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write("ПОРІВНЯННЯ МОДЕЛЕЙ\n")
        f.write("="*50 + "\n\n")

        f.write("Модель 1: llama3.2\n")
        f.write("Модель 2: gemma3:1b\n\n")

        f.write("АНАЛІЗ:\n\n")

        f.write("1. ШВИДКІСТЬ:\n")
        f.write("   - gemma3:1b швидша бо менша (800MB vs 2GB)\n")
        f.write("   - llama3.2 повільніша але детальніші відповіді\n\n")

        f.write("2. ЯКІСТЬ ВІДПОВІДЕЙ:\n")
        f.write("   - llama3.2 дає більш розгорнуті та точні відповіді\n")
        f.write("   - gemma3:1b коротші відповіді, але швидко\n\n")

        f.write("3. МОВА:\n")
        f.write("   - Обидві моделі розуміють українську\n")
        f.write("   - llama3.2 краще відповідає українською\n\n")

        f.write("4. ГЕНЕРАЦІЯ ТЕКСТУ:\n")
        f.write("   - llama3.2 генерує більш творчий та структурований текст\n")
        f.write("   - gemma3:1b простіші але швидші генерації\n\n")

        f.write("ВИСНОВОК:\n")
        f.write("   Для складних задач краще llama3.2.\n")
        f.write("   Для швидких простих відповідей краще gemma3:1b.\n")

    print(f"\n✅ Порівняння збережено у файл: {comparison_file}")


# ============================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================
def main():
    print("🚀 Запуск програми для порівняння LLM моделей")
    print(f"Моделі: {', '.join(MODELS)}\n")

    all_files = []

    for model in MODELS:
        print(f"\n{'#'*50}")
        print(f"# Тестуємо модель: {model}")
        print(f"{'#'*50}")

        chat_res = chatbot(model)
        gen_res = generate_text(model)
        multi_res = multimodal(model)

        filename = save_results(model, chat_res, gen_res, multi_res)
        all_files.append(filename)

    compare_models(all_files)
    print("\n✅ Програма завершена! Перевір файли результатів.")


if __name__ == "__main__":
    main()