"""
Streamlit интерфейс для RAG системы.
Взаимодействует с FastAPI бэкендом для поиска по документам и генерации ответов.
Поддерживает текстовый и мультимодальный поиск.
"""

import streamlit as st
import requests
import time
from PIL import Image

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
    .multimodal-result {
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("RAG Chat System")
st.markdown("Текстовый и мультимодальный поиск")

# Боковая панель
with st.sidebar:
    st.header("Информация")

    st.subheader("Режим поиска")
    search_mode = st.radio(
        "Выберите режим:",
        ["Текстовый", "Мультимодальный"],
        index=0,
    )

    if search_mode == "Мультимодальный":
        use_rerank = st.checkbox("Использовать реранкер", value=True)

    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            st.success("API сервер доступен")

            if search_mode == "Мультимодальный":
                mm_response = requests.get(
                    "http://localhost:8080/search/multimodal/health", timeout=2
                )
                if mm_response.status_code == 200:
                    mm_status = mm_response.json()
                    if mm_status.get("index_exists"):
                        st.success("Мультимодальный индекс готов")
                    else:
                        st.warning("Мультимодальный индекс не найден")
        else:
            st.error("API сервер не доступен")
    except:
        st.error("API сервер не доступен")

# Основная область
st.subheader("Введите запрос")

query = st.text_input(
    "Запрос:",
    placeholder="Введите текст запроса...",
    key="query_input",
)

col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Количество результатов", 1, 10, 5)
with col2:
    st.write("")
    search_button = st.button("Поиск", type="primary", use_container_width=True)

if search_mode == "Мультимодальный":
    col3, col4 = st.columns([3, 1])
    with col4:
        load_test_btn = st.button("Тестовые запросы", use_container_width=True)

# Загрузка тестовых запросов
if search_mode == "Мультимодальный" and "load_test_btn" in locals() and load_test_btn:
    try:
        response = requests.get("http://localhost:8080/search/multimodal/test-queries?limit=5")
        if response.status_code == 200:
            test_data = response.json()
            if test_data.get("queries"):
                st.session_state.test_queries = test_data["queries"]
                st.success(f"Загружено {len(test_data['queries'])} тестовых запросов")
    except Exception as e:
        st.error(f"Ошибка загрузки тестовых запросов: {e}")

# Отображение тестовых запросов
if "test_queries" in st.session_state and st.session_state.test_queries:
    with st.expander("Тестовые запросы"):
        for i, tq in enumerate(st.session_state.test_queries):
            if st.button(f"{tq['question'][:80]}...", key=f"test_{i}"):
                query = tq["question"]
                st.rerun()

# Обработка запроса
if search_button and query:
    with st.spinner("Поиск..."):
        try:
            start_time = time.time()

            if search_mode == "Текстовый":
                response = requests.post(
                    "http://localhost:8080/api/query",
                    json={"query": query, "top_k": top_k},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("success"):
                        elapsed_time = time.time() - start_time

                        col_res1, col_res2 = st.columns([3, 1])
                        with col_res1:
                            st.markdown(f"**Вопрос:** {result['query']}")
                        with col_res2:
                            st.caption(
                                f"Найдено: {result.get('total_found', 0)} | Время: {elapsed_time:.2f}с"
                            )

                        st.markdown("**Ответ:**")
                        st.info(result["answer"])

                        if result.get("sources"):
                            st.markdown(f"**Источники ({len(result['sources'])}):**")
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
                    else:
                        st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                else:
                    st.error(f"HTTP ошибка: {response.status_code}")

            else:
                response = requests.post(
                    "http://localhost:8080/search/multimodal",
                    json={"query": query, "n_results": top_k, "use_rerank": use_rerank},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    elapsed_time = time.time() - start_time

                    st.markdown(f"**Найдено результатов:** {result['total_found']}")
                    st.markdown(f"**Время поиска:** {result['search_time']:.2f}с")
                    st.markdown(f"**Реранкер:** {'Да' if result.get('rerank_used') else 'Нет'}")

                    st.markdown("### Результаты:")
                    for i, r in enumerate(result["results"]):
                        with st.container():
                            st.markdown(f"**Результат {i+1}** (score: {r['score']:.4f})")

                            cols = st.columns([1, 2])

                            with cols[0]:
                                if r.get("image_path"):
                                    img_path = r["image_path"].replace(
                                        "/data/", "D:/Project/reranking_multimodal_data/"
                                    )
                                    try:
                                        img = Image.open(img_path)
                                        st.image(
                                            img,
                                            caption=f"Документ {r['folder']}, Страница {r['page']}",
                                            use_container_width=True,
                                        )
                                    except:
                                        st.caption("Изображение не найдено")

                            with cols[1]:
                                st.markdown(f"**Документ:** {r['folder']}")
                                st.markdown(f"**Страница:** {r['page']}")
                                if r.get("text_preview"):
                                    with st.expander("Предпросмотр текста"):
                                        st.write(r["text_preview"])

                            st.divider()
                else:
                    st.error(f"Ошибка мультимодального поиска: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API серверу.")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

# Раздел для тестирования API
with st.expander("Тестирование API"):
    tab1, tab2 = st.tabs(["Текстовый API", "Мультимодальный API"])

    with tab1:
        st.code(
            """
curl -X POST "http://localhost:8080/api/query" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is machine learning?", "top_k": 3}'
        """
        )

        test_query = st.text_input("Запрос:", "What is AI?", key="test_text")
        if st.button("Отправить запрос", key="test_text_btn"):
            try:
                response = requests.post(
                    "http://localhost:8080/api/query", json={"query": test_query, "top_k": 2}
                )
                st.json(response.json())
            except Exception as e:
                st.error(str(e))

    with tab2:
        st.code(
            """
curl -X POST "http://localhost:8080/search/multimodal" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Which model has lowest perplexity?", "n_results": 5, "use_rerank": true}'
        """
        )

        mm_test_query = st.text_input(
            "Запрос:",
            "Which language model has the lowest Perplexity according to Table 3?",
            key="test_mm",
        )
        if st.button("Отправить запрос", key="test_mm_btn"):
            try:
                response = requests.post(
                    "http://localhost:8080/search/multimodal",
                    json={"query": mm_test_query, "n_results": 3, "use_rerank": True},
                )
                st.json(response.json())
            except Exception as e:
                st.error(str(e))

st.markdown("---")
st.caption("RAG система | Текстовый и мультимодальный режимы")
