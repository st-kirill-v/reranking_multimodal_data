import json
import sys
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def insert_before_heading(nb: dict, heading: str, new_cells: list[dict]) -> None:
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and heading in src:
            nb["cells"][i:i] = new_cells
            return
    raise SystemExit(f"Heading not found: {heading}")


def main() -> None:
    path = Path(sys.argv[1])
    nb = json.loads(path.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            src = src.replace(
                "NBA real target пока не создаётся. До join с play-by-play любые scoring labels будут proxy и не должны интерпретироваться как прогноз очков.",
                "Для NBA создаётся учебная proxy-цель `target_key_event_next_30s`: произойдёт ли в следующем 30-секундном интервале ключевое атакующее событие. Это не настоящий счёт и не shot-made label, но это допустимая учебная цель для пункта лабораторной про ключевые события на основе tracking time series.",
            )
            src = src.replace(
                "Сейчас NBA часть честно остаётся tracking-analysis блоком. Следующий обязательный шаг:",
                "Сейчас NBA часть содержит учебное обучение на proxy key-event target. Для настоящего прогноза очков следующий обязательный шаг:",
            )
            cell["source"] = src.splitlines(keepends=True)

    nba_target_cells = [
        md("### 8.1. NBA proxy target для обучения LSTM"),
        code(
            r"""
nba_tracking_df["intensity_score"] = (
    nba_tracking_df["avg_distance"].rank(pct=True)
    + nba_tracking_df["std_distance"].rank(pct=True)
    + nba_tracking_df["spread_x"].rank(pct=True)
    + nba_tracking_df["spread_y"].rank(pct=True)
) / 4

strict_key_event = (
    (
        (nba_tracking_df["ball_hoop_dist"] <= 10)
        | (nba_tracking_df["min_player_hoop_dist"] <= 6)
    )
    & (
        (nba_tracking_df["low_shot_clock"] == 1)
        | (nba_tracking_df["players_near_hoop"] >= 2)
        | (nba_tracking_df["intensity_score"] >= nba_tracking_df["intensity_score"].quantile(0.85))
    )
)
nba_tracking_df["is_key_event_proxy"] = strict_key_event.astype(int)
nba_tracking_df["target_key_event_next_30s"] = (
    nba_tracking_df
    .sort_values(["match_id", "interval_id"])
    .groupby("match_id")["is_key_event_proxy"]
    .shift(-1)
)
nba_supervised_df = nba_tracking_df.dropna(subset=["target_key_event_next_30s"]).copy()
nba_supervised_df["target_key_event_next_30s"] = nba_supervised_df["target_key_event_next_30s"].astype(int)

print("NBA supervised proxy dataframe:", nba_supervised_df.shape)
print("Target distribution:")
display(nba_supervised_df["target_key_event_next_30s"].value_counts().sort_index().to_frame("count"))
display(nba_supervised_df["target_key_event_next_30s"].value_counts(normalize=True).sort_index().to_frame("share"))
display(nba_supervised_df.head(30))

print(
    "Важно: target_key_event_next_30s — это proxy key event, "
    "а не реальный shot_made/points label. Он закрывает учебный этап обучения на NBA Tracking, "
    "но в выводах нужно честно указать ограничение."
)
            """
        ),
    ]
    insert_before_heading(nb, "## 9. Анализ после предобработки Football", nba_target_cells)

    sequence_cells = [
        md("### 13.1. Последовательности NBA для proxy key-event LSTM"),
        code(
            r"""
NBA_FEATURES = [
    "interval_id",
    "period",
    "avg_distance",
    "std_distance",
    "spread_x",
    "spread_y",
    "ball_x",
    "ball_y",
    "shot_clock_mean",
    "game_clock_mean",
    "players_count",
    "ball_hoop_dist",
    "min_player_hoop_dist",
    "players_near_hoop",
    "low_shot_clock",
    "intensity_score",
]
NBA_TARGET = "target_key_event_next_30s"

n_train_ids, n_val_ids, n_test_ids = split_match_ids(nba_supervised_df)
nba_train = nba_supervised_df[nba_supervised_df["match_id"].isin(n_train_ids)].copy()
nba_val = nba_supervised_df[nba_supervised_df["match_id"].isin(n_val_ids)].copy()
nba_test = nba_supervised_df[nba_supervised_df["match_id"].isin(n_test_ids)].copy()

nba_train_s, nba_val_s, nba_test_s, nba_scaler = scale_split(nba_train, nba_val, nba_test, NBA_FEATURES)
X_train_nba, y_train_nba, meta_train_nba = make_sequences(
    nba_train_s, NBA_FEATURES, NBA_TARGET, time_col="interval_id", time_steps=TIME_STEPS
)
X_val_nba, y_val_nba, meta_val_nba = make_sequences(
    nba_val_s, NBA_FEATURES, NBA_TARGET, time_col="interval_id", time_steps=TIME_STEPS
)
X_test_nba, y_test_nba, meta_test_nba = make_sequences(
    nba_test_s, NBA_FEATURES, NBA_TARGET, time_col="interval_id", time_steps=TIME_STEPS
)

print("NBA X_train:", X_train_nba.shape, "X_val:", X_val_nba.shape, "X_test:", X_test_nba.shape)
print("NBA train target distribution:", pd.Series(y_train_nba.astype(int)).value_counts().sort_index().to_dict())
            """
        ),
    ]
    insert_before_heading(nb, "## 14. Построение LSTM-моделей", sequence_cells)

    train_cells = [
        md("### 15.1. Обучение NBA LSTM на proxy key-event target"),
        code(
            r"""
nba_lstm_model = build_lstm_binary((TIME_STEPS, len(NBA_FEATURES)), "lstm_nba_proxy_key_event")
print("Training NBA LSTM for target_key_event_next_30s")
nba_history = nba_lstm_model.fit(
    X_train_nba,
    y_train_nba,
    epochs=EPOCHS,
    batch_size=128,
    validation_data=(X_val_nba, y_val_nba),
    class_weight=compute_weights(y_train_nba),
    verbose=1,
)
            """
        ),
    ]
    insert_before_heading(nb, "## 16. Метрики", train_cells)

    metric_cells = [
        md("### 16.1. Метрики NBA proxy key-event модели"),
        code(
            r"""
nba_prob = nba_lstm_model.predict(X_test_nba, verbose=0).ravel()
nba_metrics = evaluate_binary(y_test_nba.astype(int), nba_prob, "LSTM_NBA_target_key_event_next_30s")
display(pd.DataFrame([nba_metrics]))

nba_pred = (nba_prob >= 0.5).astype(int)
display(pd.DataFrame(
    confusion_matrix(y_test_nba.astype(int), nba_pred),
    index=["true_0", "true_1"],
    columns=["pred_0", "pred_1"],
))

majority_class = int(pd.Series(y_train_nba.astype(int)).mode()[0])
majority_prob = np.full_like(y_test_nba.astype(float), fill_value=majority_class, dtype=float)
majority_metrics = evaluate_binary(y_test_nba.astype(int), majority_prob, "NBA_majority_baseline")
display(pd.DataFrame([majority_metrics]))
            """
        ),
    ]
    insert_before_heading(nb, "## 17. Визуализация", metric_cells)

    viz_cells = [
        md("### 17.1. Визуализация NBA proxy key-event обучения"),
        code(
            r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(nba_history.history["loss"], label="train")
axes[0].plot(nba_history.history["val_loss"], label="val")
axes[0].set_title("NBA LSTM loss")
axes[0].legend()

axes[1].plot(nba_history.history["accuracy"], label="train")
axes[1].plot(nba_history.history["val_accuracy"], label="val")
axes[1].set_title("NBA LSTM accuracy")
axes[1].legend()

sns.heatmap(confusion_matrix(y_test_nba.astype(int), nba_pred), annot=True, fmt="d", cmap="Greens", ax=axes[2])
axes[2].set_title("NBA confusion matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True")

plt.tight_layout()
plt.show()

precision_nba, recall_nba, _ = precision_recall_curve(y_test_nba.astype(int), nba_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall_nba, precision_nba)
plt.title("NBA proxy key-event PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()
            """
        ),
    ]
    insert_before_heading(nb, "## 18. Итоговые выводы", viz_cells)

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Added NBA proxy training to {path}")


if __name__ == "__main__":
    main()
