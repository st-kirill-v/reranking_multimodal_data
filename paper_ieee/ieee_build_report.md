# IEEE Build Report

Дата: 2026-06-19.

## Сгенерированные файлы

- `paper_ieee/paper.tex`
- `paper_ieee/references.bib`
- `paper_ieee/figures/`
- `paper_ieee/tables/`
- `paper_ieee/pdf/`

## Статистика проекта

- Количество страниц PDF: не определено, PDF не собран в текущем окружении.
- Количество таблиц: 8.
- Количество рисунков: 1.
- Количество источников: 21.

## Сборка PDF

PDF `paper_ieee/pdf/article_ieee.pdf` не создан, потому что в PATH не найдены LaTeX-инструменты `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `tectonic` и `bibtex`.

Рекомендуемая команда в окружении с установленным TeX Live или MiKTeX:

```powershell
cd paper_ieee
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
Move-Item paper.pdf pdf/article_ieee.pdf -Force
```

## Проверки оформления

- Использован IEEE conference class: `\documentclass[conference]{IEEEtran}`.
- Abstract оформлен через окружение `abstract`.
- Keywords оформлены через окружение `IEEEkeywords`.
- Библиография вынесена в `references.bib` и подключена через BibTeX.
- Markdown-таблицы перенесены в LaTeX как `table*` без удаления строк или чисел.
- Mermaid-схема pipeline перенесена в TikZ-рисунок с тем же содержанием.

## Предупреждения компиляции

Компиляция не запускалась из-за отсутствия LaTeX toolchain. Потенциальные предупреждения после установки LaTeX:

- статья существенно длиннее типичного IEEE conference paper;
- широкие таблицы оформлены через `table*` и `\resizebox{\textwidth}{!}{...}`, поэтому могут требовать ручной версточной донастройки;
- русскоязычный IEEE-проект требует LaTeX с поддержкой `T2A`, `babel` и кириллических шрифтов;
- TikZ-схема pipeline заменяет исходный Mermaid-блок как LaTeX-рисунок с тем же содержанием.

## Рекомендации по сокращению при необходимости

Итоговый текст по объему, вероятно, превышает типичный IEEE conference paper. Автоматические сокращения не выполнялись. Если площадка потребует лимит 4-6 страниц, сокращать следует вручную и только после согласования: Related Work, большие сравнительные таблицы и подробные экспериментальные интерпретации.
