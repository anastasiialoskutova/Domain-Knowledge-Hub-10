# Схема витягу знань та інструкція розмітки (Результат 1)

## 1. Типи сутностей (6 типів)
1. ALGORITHM — моделі рекомендацій і скорингу (Hybrid RRF, SASRec, Personalized PageRank). Контрприклад: «система» без назви.
2. METRIC — показники якості та ризику (LTV, CTR, Recall@5, Refund Rate). Контрприклад: «прибуток».
3. FIN_PRODUCT — фінансові продукти e-commerce (BNPL-розстрочка, Кешбек-картка, Кредитний ліміт). Контрприклад: «ноутбук».
4. SEGMENT — сегменти клієнтів (VIP-клієнти, Cold Start, Loyal-клієнти, Prime-позичальники, At-Risk).
5. CODE_STD — коди і стандарти (MCC 5411, MCC 5732, MCC 6012, PCI DSS).
6. THRESHOLD — числові пороги та вікна (25%, 90 днів, 500 грн, 0.05).

## 2. Типи відношень (6 типів)
1. OPTIMIZES(ALGORITHM -> METRIC)
2. APPLIES_TO(FIN_PRODUCT -> SEGMENT)
3. GOVERNED_BY(FIN_PRODUCT|CODE_STD -> CODE_STD)
4. HAS_THRESHOLD(METRIC|ALGORITHM -> THRESHOLD)
5. MITIGATES(ALGORITHM -> SEGMENT)
6. USES_FEATURE(ALGORITHM -> METRIC|CODE_STD)

## 3. Правила розмітки та JSON-схема
1. Розмічаються лише факти з точною цитатою (evidence) та офсетами [start_char, end_char].
2. Абревіатури не дублюються в одному реченні.
3. Заперечення не розмічаються як позитивні зв`язки.
4. Пороги прив`язуються до найближчої метрики через HAS_THRESHOLD.
5. Перевіряється напрямок (subject -> object).

## 4. Пілотна розмітка (5 фрагментів) та 2 уточнення
1. Розділено RFM-скоринг (ALGORITHM) і числовий LTV (METRIC).
2. Закріплено збереження одиниць виміру у THRESHOLD (25%, 90 днів).
