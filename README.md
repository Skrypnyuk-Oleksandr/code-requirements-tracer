# Requirements Tracing with LLM

Автоматичне трасування вимог до програмного коду з використанням великих мовних моделей (LLM).  
Репозиторій містить реалізацію методу для виявлення семантичних зв’язків між текстовими вимогами та фрагментами коду, а також приклади експериментів.

## Структура репозиторію
requirements-tracing-llm/
│
├─ CODE1.py
├─ CODE2.py
├─ CODE3.py
├─ CODE4.py
│
├─ MSRCaseStudy/ # кейс-дослідження та дані прикладів
│ └─ ...
│
├─ emb_cache/ # кеш ембедінгів для тексту/коду
│ └─ ...
│
├─ emb_cache_0/ # резервний кеш ембедінгів
│ └─ ...
│
├─ README.md # цей файл
├─ requirements.txt # залежності Python
└─ LICENSE # ліцензія

## Встановлення

1. Клонувати репозиторій:

```bash
git clone https://github.com/Skrypnyuk-Oleksandr/requirements-tracing-llm.git
cd requirements-tracing-llm
pip install -r requirements.txt

