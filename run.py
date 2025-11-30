import sys
import os

# Добавляем src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Теперь импортируем правильно
from src.core.rag import rag_engine
from src.api.server import app

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
