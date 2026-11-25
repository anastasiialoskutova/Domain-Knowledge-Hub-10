import json
import os
import re
from datetime import datetime, timedelta
import random

random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

TOPICS = [
    ("DOC-01", "RFM-сегментація клієнтів у фінтех e-commerce", "methodology", "public", "https://hub.internal.ua/kb/rfm-segmentation"),
    ("DOC-02", "Колаборативна фільтрація на основі матриці транзакцій", "algorithm", "public", "https://hub.internal.ua/kb/cf-transactions"),
    ("DOC-03", "Контентні рекомендації за профілем фінансових продуктів", "algorithm", "public", "https://hub.internal.ua/kb/content-based-fin"),
    ("DOC-04", "Проблема холодного старту (Cold Start) для нових покупців", "methodology", "public", "https://hub.internal.ua/kb/cold-start"),
    ("DOC-05", "Скоринг платоспроможності при рекомендації BNPL-розстрочки", "risk_policy", "restricted", "https://hub.internal.ua/kb/bnpl-scoring"),
    ("DOC-06", "Поведінкові патерни: аналіз покинутих кошиків та мікроконверсій", "analytics", "internal", "https://hub.internal.ua/kb/cart-abandonment"),
    ("DOC-07", "Метрики якості рекомендацій: Precision@K, Recall@K, nDCG, MRR", "evaluation", "public", "https://hub.internal.ua/kb/recsys-metrics"),
    ("DOC-08", "Персоналізований PageRank на дводольному графі Користувач-Товар", "algorithm", "public", "https://hub.internal.ua/kb/ppr-graph"),
    ("DOC-09", "Розрахунок LTV (Lifetime Value) для ранжування пропозицій", "finance", "internal", "https://hub.internal.ua/kb/ltv-ranking"),
    ("DOC-10", "Рекомендації кешбек-категорій на базі історії MCC-кодів", "finance", "public", "https://hub.internal.ua/kb/mcc-cashback"),
    ("DOC-11", "Захист персональних даних та PCI DSS у системах рекомендацій", "compliance", "restricted", "https://hub.internal.ua/kb/pci-dss-recsys"),
    ("DOC-12", "Сезонність та часове згасання ваг історії покупок (Time Decay)", "algorithm", "public", "https://hub.internal.ua/kb/time-decay"),
    ("DOC-13", "Антифрод-фільтрація аномальних сплесків купівельної активності", "risk_policy", "restricted", "https://hub.internal.ua/kb/antifraud-filter"),
    ("DOC-14", "Гібридні рекомендаційні архітектури: двохетапний Retrieval + Ranking", "architecture", "public", "https://hub.internal.ua/kb/two-stage-recsys"),
    ("DOC-15", "Векторні ембедінги користувачів (User2Vec) за послідовністю чеків", "algorithm", "public", "https://hub.internal.ua/kb/user2vec"),
    ("DOC-16", "A/B тестування рекомендаційних моделей та розрахунок GMV uplift", "evaluation", "internal", "https://hub.internal.ua/kb/ab-testing-gmv"),
    ("DOC-17", "Крос-продажі (Cross-sell) та апсейл (Up-sell) на сторінці чекауту", "business_logic", "public", "https://hub.internal.ua/kb/cross-upsell"),
    ("DOC-18", "Управління диверсифікацією та новизною (Novelty & Coverage)", "evaluation", "public", "https://hub.internal.ua/kb/diversity-novelty"),
    ("DOC-19", "Політика кредитних лімітів при персоналізації товарної видачі", "risk_policy", "restricted", "https://hub.internal.ua/kb/credit-limits"),
    ("DOC-20", "Онтологія товарів та фінансових тригерів у e-commerce", "ontology", "public", "https://hub.internal.ua/kb/ecommerce-ontology"),
    ("DOC-21", "Сесійні рекомендації на основі RNN та Transformer (SASRec)", "algorithm", "public", "https://hub.internal.ua/kb/sasrec-session"),
    ("DOC-22", "Оптимізація маржинальності (Margin-aware Recommendation)", "finance", "internal", "https://hub.internal.ua/kb/margin-aware"),
    ("DOC-23", "Повернення товарів (Refund rate) як штрафний сигнал у навчанні", "analytics", "internal", "https://hub.internal.ua/kb/refund-penalty"),
    ("DOC-24", "Інтеграція історії пошукових запитів із транзакційним профілем", "architecture", "public", "https://hub.internal.ua/kb/search-tx-fusion"),
]

SECTIONS_TEMPLATES = [
    ("1. Визначення та бізнес-контекст",
     "У сучасній електронній комерції з інтегрованими фінансовими сервісами механізм «{title}» дозволяє адаптувати товарну та кредитну видачу під реальну купівельну спроможність клієнта. Основна мета полягає у максимізації конверсії та середнього чека (AOV) без перевищення допустимого ризик-профілю транзакції. Враховуються частота покупок, середній інтервал між замовленнями та чутливість до знижок."),
    ("2. Математична формалізація та ознаки",
     "Для побудови ознак використовуються агрегати транзакційної історії за вікнами 7, 30 та 90 днів: сумарний оборот, частка оплат кредитними коштами, співвідношення переглядів до покупок та індекс категорійної концентрації Герфіндаля-Хіршмана. Вага кожної історичної взаємодії масштабується експоненційним коефіцієнтом згасання exp(-lambda * delta_t), де delta_t вимірюється у добах."),
    ("3. Вплив фінансових обмежень та профілю користувача",
     "Перед формуванням фінального списку рекомендацій система перевіряє доступний кредитний ліміт, історію повернень товарів (refund rate) та поточний статус бонусного рахунку. Якщо частка повернень користувача перевищує 25%, товари з високою вартістю зворотної логістики знижуються у рейтингу. Для клієнтів сегмента VIP пріоритет надається преміальним позиціям із вищою маржинальністю."),
    ("4. Архітектурна реалізація та обробка даних",
     "Потокова обробка подій клікстріму та чеків здійснюється через черги повідомлень із подальшим оновленням векторного профілю у сховищі ознак (Feature Store) із затримкою до 150 мс. Пакетний перерахунок матриць схожості виконується щодобово. Усі чутливі платіжні ідентифікатори токенізуються до етапу побудови ембедінгів."),
    ("5. Оцінка ефективності та типові помилки",
     "Ключовими показниками успішності впровадження є приріст чистого доходу на користувача (ARPU), зниження частки прострочених BNPL-платежів та утримання показника Recall@5 на рівні не нижче 0.68. Типовою помилкою є надмірна оптимізація під кліки (CTR), що призводить до рекомендації дешевих аксесуарів замість цільових маржинальних товарів."),
    ("6. Практичні рекомендації для інтеграції в Knowledge Hub",
     "Під час індексації документів цього типу необхідно зберігати прив'язку до версії ризик-політики та рівня конфіденційності. Запити, що стосуються кредитного скорингу або внутрішніх порогів відсікання, повинні фільтруватися за метаданими sensitivity ще до етапу векторного пошуку."),
    ("7. Приклад розрахунку параметрів на реальному кейсі",
     "Розглянемо покупця з історією 12 замовлень за квартал із середнім чеком 2450 грн та своєчасним погашенням двох розстрочок. Модель присвоює йому коефіцієнт довіри 0.89, що відкриває доступ до персоналізованих добірок електроніки та побутової техніки з нульовою комісією на 6 платежів."),
    ("8. Взаємозв'язок із суміжними модулями системи",
     "Модуль «{title}» безпосередньо обмінюється даними з графом знань товарів (SKU-онтологією), сервісом динамічного ціноутворення та двигуном антифрод-моніторингу. Узгодженість схем даних гарантується через єдиний реєстр метаданих та автоматичну валідацію перед оновленням індексу."),
    ("9. Контрольні правила валідації якості даних",
     "Перед завантаженням у пайплайн перевіряється відсутність дублікатів транзакцій, коректність валютних курсів на дату чека та заповненість категорійних атрибутів. Фрагменти з порожнім полем source_url або некоректною датою відхиляються на етапі ETL.")
]

def clean_noise(raw_text: str) -> str:
    """Очищення від навігаційних блоків, HTML-тегів та службових колонтитулів."""
    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = re.sub(r"\[Навігація:.*?\]", "", text)
    text = re.sub(r"©\s*2026\s*Domain Knowledge Hub.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

corpus = []
base_date = datetime(2026, 9, 1)

for idx, (doc_id, title, doc_type, sensitivity, url) in enumerate(TOPICS):
    doc_date = (base_date + timedelta(days=idx * 3)).strftime("%Y-%m-%d")
    for s_idx, (sec_title, sec_body) in enumerate(SECTIONS_TEMPLATES, start=1):
        # Імітація сирого тексту з навігаційним шумом та колонтитулом
        raw_chunk = f"[Навігація: Головна > Каталог знань > {doc_id}] <p>{sec_body.format(title=title)}</p> © 2026 Domain Knowledge Hub | Всі права захищено"
        cleaned_text = clean_noise(raw_chunk)
        
        # Додаткове підвищення sensitivity для секцій про ліміти/скоринг
        chunk_sens = sensitivity
        if "кредит" in cleaned_text.lower() or "скоринг" in cleaned_text.lower() or "антифрод" in cleaned_text.lower():
            chunk_sens = "restricted"

        record = {
            "chunk_id": f"{doc_id}-C{s_idx:02d}",
            "doc_id": doc_id,
            "title": title,
            "section": sec_title,
            "type": doc_type,
            "date": doc_date,
            "sensitivity": chunk_sens,
            "source_url": url,
            "text": cleaned_text
        }
        corpus.append(record)

with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Згенеровано документів: {len(TOPICS)}")
print(f"Згенеровано фрагментів (чанків): {len(corpus)}")