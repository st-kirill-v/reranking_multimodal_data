"""
Streamlit интерфейс для RAG системы.
Взаимодействует с FastAPI бэкендом для поиска по документам и генерации ответов.
"""

import streamlit as st
import requests
import time

# Настройка страницы
st.set_page_config(page_title="RAG Chat System", layout="wide")

# Стили для интерфейса
st.markdown(
    """
    <style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .source-box {
        padding: 10px;
        margin: 5px 0;
        background-color: #f0f2f6;
        border-radius: 5px;
        border-left: 4px solid #4a90e2;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("RAG Chat System")
st.markdown("Модульный поиск с генерацией ответов")

# Боковая панель с информацией о системе
with st.sidebar:
    st.header("Системная информация")

    try:
        response = requests.get("http://localhost:8000/info", timeout=5)
        if response.status_code == 200:
            info = response.json()

            if "llm_generator" in info:
                st.metric("LLM Генератор", info["llm_generator"].get("name", "DialoGPT"))

            if "active_modules" in info:
                st.write(f"Активные модули: {len(info['active_modules'])}")
                for module in info["active_modules"]:
                    st.write(f"- {module}")
        else:
            st.warning("Не удалось получить информацию о системе")
    except:
        st.error("API сервер не доступен")

# Основная область чата
st.subheader("Задайте вопрос")

query = st.text_input(
    "Вопрос (на английском для SQuAD датасета):",
    placeholder="Пример: What is artificial intelligence?",
    key="query_input",
)

col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Количество источников", 1, 5, 3)
with col2:
    st.write("")
    search_button = st.button("Поиск и генерация", type="primary", use_container_width=True)

# Обработка запроса пользователя
if search_button and query:
    with st.spinner("Идет поиск по документам и генерация ответа..."):
        try:
            start_time = time.time()

            response = requests.post(
                "http://localhost:8000/api/query", json={"query": query, "top_k": top_k}, timeout=30
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                if result.get("success"):
                    st.success(f"Ответ получен за {elapsed_time:.2f} сек")
                    st.markdown(f"**Вопрос:** {result['query']}")
                    st.markdown("**Ответ:**")
                    st.info(result["answer"])

                    if result.get("sources"):
                        st.markdown(f"**Использованные источники ({len(result['sources'])}):**")

                        for i, source in enumerate(result["sources"]):
                            with st.expander(
                                f"Источник {i+1} (релевантность: {source.get('score', 0):.3f})"
                            ):
                                if "preview" in source:
                                    st.write(source["preview"])
                                elif "content" in source:
                                    content = source["content"]
                                    if len(content) > 300:
                                        st.write(content[:300] + "...")
                                    else:
                                        st.write(content)

                    st.caption(
                        f"Найдено документов: {result.get('total_found', 0)} | Время: {elapsed_time:.2f} сек"
                    )
                else:
                    st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
            else:
                st.error(f"HTTP ошибка: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API серверу.")
        except requests.exceptions.Timeout:
            st.error("Таймаут запроса.")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")

# Раздел для тестирования API
with st.expander("Ручной тест API"):
    st.markdown(
        """
    Пример тестирования через curl:

    curl -X POST "http://localhost:8000/api/query" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "What is machine learning?", "top_k": 3}'
    """
    )

    test_query = st.text_input("Тестовый запрос:", "What is AI?")
    if st.button("Протестировать"):
        try:
            response = requests.post(
                "http://localhost:8000/api/query", json={"query": test_query, "top_k": 2}
            )
            st.json(response.json())
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.caption("RAG система с модульной архитектурой")
