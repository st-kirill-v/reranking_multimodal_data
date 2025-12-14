# test_tensorflow_e5.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print("🧪 Тестирую E5 с TensorFlow...")

# 1. Проверяем TensorFlow
try:
    import tensorflow as tf

    print(f"✅ TensorFlow версия: {tf.__version__}")
    print(f"   GPU доступен: {len(tf.config.list_physical_devices('GPU')) > 0}")
except:
    print("❌ TensorFlow не установлен")
    exit()

# 2. Проверяем transformers
try:
    import transformers

    print(f"✅ Transformers версия: {transformers.__version__}")
except:
    print("❌ Transformers не установлен")
    exit()

# 3. Создаем и тестируем E5 модуль
from src.core.modules.e5_module import E5Module

try:
    e5 = E5Module(name="tf_e5", model_path="./models/e5/e5-small-v2", bm25_module_name="bm25")
    print(f"✅ E5Module создан. Бэкенд: {e5.get_info()['backend']}")

    # Тестовое кодирование
    test_query = "What is artificial intelligence?"
    embedding = e5._encode_text(test_query, is_query=True)
    print(f"✅ Эмбеддинг создан. Размерность: {embedding.shape}")

    # Тестовый поиск
    results = e5.search(test_query, top_k=3)
    print(f"✅ Поиск выполнен. Результатов: {len(results)}")

    if results:
        for i, r in enumerate(results):
            print(f"  {i+1}. Score: {r['score']:.3f} | Backend: {r.get('backend', 'unknown')}")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()
