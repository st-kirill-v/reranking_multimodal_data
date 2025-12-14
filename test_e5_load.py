# test_e5_load.py
import os
import torch
from transformers import AutoTokenizer, AutoModel


def test_model_loading():
    model_path = "./models/e5/e5-small-v2"

    print("🧪 Тестирую загрузку E5 модели...")
    print(f"📁 Путь: {model_path}")

    # Проверка файлов
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
    for file in required_files:
        filepath = os.path.join(model_path, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024 / 1024
            print(f"✅ {file}: {size:.1f} MB")
        else:
            print(f"❌ {file}: отсутствует")
            return False

    # Пробуем загрузить модель
    try:
        print("\n🔧 Загружаю токенизатор...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("✅ Токенизатор загружен")

        print("🔧 Загружаю модель...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Устройство: {device}")

        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()

        print("✅ Модель загружена успешно!")

        # Тестовое кодирование
        print("\n🧪 Тестовое кодирование...")
        texts = ["Hello world", "What is artificial intelligence?"]

        # E5 требует префиксы
        texts_with_prefix = [f"query: {text}" for text in texts]

        inputs = tokenizer(
            texts_with_prefix, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        print(f"✅ Кодирование успешно!")
        print(f"   Размерность эмбеддингов: {embeddings.shape}")
        print(f"   Устройство: {device}")

        return True

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False


if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Всё готово! Теперь запустите тест каскадного поиска:")
        print("python test_rag.py")
    else:
        print("\n⚠️  Есть проблемы с моделью")
