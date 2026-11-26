import os
from fpdf import FPDF

# Знаходимо системний шрифт macOS із підтримкою кирилиці
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/Library/Fonts/Arial.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

class WhitepaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("MainFont", "", 8.5)
            self.set_text_color(110, 110, 110)
            self.cell(0, 8, "Domain Knowledge Hub — Whitepaper (ЛР 16) | ЛНУ ім. Івана Франка", align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("MainFont", "", 8.5)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Сторінка {self.page_no()} / {{nb}}", align="C")

pdf = WhitepaperPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_font("MainFont", "", FONT_PATH)
pdf.add_font("MainFont", "B", FONT_PATH)

def add_section(title, paragraphs):
    pdf.set_font("MainFont", "B", 13)
    pdf.set_text_color(20, 55, 100)
    pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("MainFont", "", 10.5)
    pdf.set_text_color(25, 25, 25)
    for p in paragraphs:
        pdf.multi_cell(0, 5.8, p)
        pdf.ln(2.5)

# Сторінка 1: Титульна + Розділ 1
pdf.add_page()
pdf.set_font("MainFont", "B", 17)
pdf.set_text_color(15, 45, 90)
pdf.multi_cell(0, 8.5, "ТЕХНІЧНА ЗАПИСКА (WHITEPAPER)\nГібридна рекомендаційна система на основі графа знань для електронної комерції та фінтех-продуктів", align="C")
pdf.ln(4)
pdf.set_font("MainFont", "", 10)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 6, "Проєкт: Domain Knowledge Hub | Виконала: Лоскутова А. А. (група Феі-43с) | Перевірив: Патинко А. М. | Львів 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)

add_section("1. Індустріальна проблема, цільові користувачі та бізнес-цінність", [
    "Сучасні платформи електронної комерції та фінансових сервісів стикаються з проблемою низької конверсії персональних пропозицій (BNPL-розстрочка, кешбек-картки, кредитні ліміти) через фрагментацію даних між поведінковою аналітикою, довідниками MCC-кодів та нормативними обмеженнями ризик-менеджменту.",
    "Традиційні моделі колаборативної фільтрації працюють як «чорна скринька»: вони не здатні пояснити причину рекомендації, страждають від холодного старту для нових клієнтів і можуть пропонувати продукти без перевірки актуальності тарифів чи регуляторних вимог (PCI DSS).",
    "Цільові користувачі системи: (1) бізнес-аналітики фінтех-продуктів; (2) інженери рекомендаційних систем; (3) спеціалісти з ризик-менеджменту та комплаєнсу; (4) кінцеві клієнти e-commerce, які отримують прозорі фінансові підказки."
])

# Сторінка 2: Архітектура Hub
pdf.add_page()
add_section("2. Архітектура системи Domain Knowledge Hub", [
    "Система побудована як багаторівневий гібридний конвеєр, що поєднує семантичний граф знань (RDF/OWL, Turtle), векторний пошук по нормативних документах, аналітичну вітрину SQLite OLAP, MCP-інструменти LLM-агента та гібридний ранжувальник Hybrid RRF.",
    "Рівень 1 (Інженерія знань та онтологія, ЛР 1-7): формалізує сутності FinProduct, Algorithm, Metric, Segment, Standard та зв'язки optimizes, governedBy, appliesTo із фіксацією джерел через prov:wasDerivedFrom.",
    "Рівень 2 (Контроль якості та CI Gate, ЛР 14): забезпечує 100% валідність даних через pySHACL, SPARQL-інваріанти та дедуплікацію сутностей перед потраплянням у продакшн-граф.",
    "Рівень 3 (Агентний доступ та рекомендації, ЛР 15-16): надає безпечний read-only доступ через MCP-сервер і виконує гібридне ранжування (Graph Paths + TF-IDF Embeddings) із перевіркою пояснень на базі шляхів графа."
])

# Сторінка 3: Знання, походження та зведені метрики якості (ЛР 3-15)
pdf.add_page()
add_section("3. Організація знань, походження (Provenance) та метрики якості (ЛР 3-15)", [
    "Кожен факт у сховищі знань супроводжується метаданими походження prov:wasDerivedFrom (ідентифікатори DOC-01..DOC-18), датою оновлення hub:updatedAt та механізмом збереження історичних альтернатив hub:hasAlternative.",
    "Таблиця 1. Зведена динаміка метрик якості Domain Knowledge Hub (ЛР 11 - ЛР 16):",
    "  * Повнота та валідність графа (ЛР 14): зросла з 83.3% (Baseline) до 100.0% (Clean KG) за 11 правилами SHACL/SPARQL.\n"
    "  * Дедуплікація сутностей (ЛР 14): Precision = 1.000, Recall = 0.800, F1 = 0.889 на ізольованій вибірці test (n=50).\n"
    "  * Якість відповідей RAG-системи (ЛР 11 -> ЛР 14): Faithfulness зріс із 0.812 до 0.967 (+0.155), Context Recall — із 0.767 до 0.933 (+0.166).\n"
    "  * Надійність та безпека LLM-агента (ЛР 15): Success Rate = 100.0% (10/10 сценаріїв), блокування атак Red Teaming = 100.0% (6/6 атак).\n"
    "  * Офлайн-оцінка рекомендацій (ЛР 16, n=60): Hybrid RRF досяг максимального балансу Recall@5, nDCG@10 та Coverage при 100% точності пояснень."
])

# Сторінка 4: Графік результатів ЛР 16
pdf.add_page()
add_section("4. Експериментальна оцінка рекомендаційного модуля (ЛР 16)", [
    "Для уникнення витоку даних із майбутнього застосовано суворе часове розбиття (Temporal Split): навчальна вибірка охоплює взаємодії 2026-10-10..11, тестова — 2026-11-15..16 по 60 користувачах (240 взаємодій).",
    "Рисунок 1 демонструє перевагу гібридного ранжування Hybrid RRF (w_graph=0.65, w_emb=0.35) над базовим Popularity Baseline та ізольованими підходами за метриками Recall@5, nDCG@10 та Coverage:"
])
if os.path.exists("figures/reco_metrics.png"):
    pdf.image("figures/reco_metrics.png", x=15, w=180)
pdf.ln(3)
pdf.set_font("MainFont", "", 9.5)
pdf.cell(0, 6, "Рисунок 1 — Порівняння рекомендаційних моделей на часовому розбитті (train -> test, n=60)", align="C", new_x="LMARGIN", new_y="NEXT")

# Сторінка 5: Ризики за EU AI Act та NIST AI RMF
pdf.add_page()
add_section("5. Управління ризиками та відповідність EU AI Act і NIST AI RMF 1.0", [
    "Відповідно до Регламенту ЄС про штучний інтелект (EU AI Act, Regulation 2024/1689), системи оцінювання кредитоспроможності належать до систем високого ризику (High-Risk AI, Додаток III). Наш модуль виконує роль інформаційно-рекомендаційного асистента (Limited Risk / Transparency Obligations, ст. 50) і не приймає автоматичних рішень про відмову у кредиті.",
    "Таблиця 2. Матриця ризиків та заходів пом'якшення за NIST AI RMF 1.0 (Govern, Map, Measure, Manage):",
    "  1. Прозорість і пояснюваність (Explainability): кожна рекомендація супроводжується верифікованим шляхом графа (verify_explanation_grounding = 100%).\n"
    "  2. Конфіденційність та мінімізація даних (Privacy): MCP-інструменти працюють у режимі read-only без передачі персональних даних (PII) у зовнішні LLM.\n"
    "  3. Стійкість до атак (Security / Red Teaming): санітизація непрямих ін'єкцій та ліміт кроків max_steps=5 забезпечують 100% блокування тестових атак.\n"
    "  4. Нагляд людини (Human-in-the-Loop): будь-які зміни фінансових порогів проходять через контур propose_fact із ручним схваленням."
])

# Сторінка 6: Економіка, вартість, обмеження та наступні кроки
pdf.add_page()
add_section("6. Економіка рішення, обмеження та план розвитку", [
    "Економічна ефективність: завдяки компактному графу знань, локальним TF-IDF embeddings та кешуванню SPARQL-запитів середня вартість обробки одного агентного запиту становить $0.0012 при затримці < 20 мс, що у 15 разів дешевше за виклик великих хмарних моделей без графового контексту.",
    "Обмеження системи: (1) залежність повноти графа від регулярності оновлення нормативних документів; (2) використання статичних TF-IDF векторів замість динамічних послідовних ембедингів (SASRec) для довгих сесій; (3) синтетичний характер частини поведінкових взаємодій.",
    "Наступні кроки: інтеграція реальних потокових подій Apache Kafka у графову БД Neo4j, впровадження онлайн A/B-тестування та розширення SHACL-профілів для автоматичної перевірки змін у законодавстві НБУ."
])

# Сторінка 7: Використання генеративного ШІ та список літератури
pdf.add_page()
add_section("7. Декларація використання генеративного ШІ", [
    "У межах проєкту генеративна модель Gemini (temperature=0.0, seed=42) використовувалася як LLM-експерт для: (а) генерації чернеток онтологічних правил та синонімів; (б) формування текстових пояснень рекомендацій на основі шляхів графа; (в) моделювання атак Red Teaming.",
    "Усі виходи моделі проходили обов'язкову багаторівневу верифікацію: автоматичну перевірку шляхів графа (V1/V2), рецензування інженером знань (V3) та офлайн-оцінювання на часовому розбитті (V4). Частка підтверджених тверджень становить 75.0% (15/20), відхилено 25.0% (5/20) необґрунтованих чернеток."
])
add_section("8. Список використаних джерел (8 джерел)", [
    "1. Ricci F., Rokach L., Shapira B. Recommender Systems Handbook. — 2nd ed. — Springer, 2015. — 1003 p.\n"
    "2. Aggarwal C. C. Recommender Systems: The Textbook. — Springer, 2016. — 498 p.\n"
    "3. Regulation (EU) 2024/1689 of the European Parliament and of the Council (EU AI Act). — Official Journal of the EU, 2024.\n"
    "4. NIST AI Risk Management Framework (AI RMF 1.0). — National Institute of Standards and Technology, NIST AI 100-1, 2023.\n"
    "5. Yao S. et al. ReAct: Synergizing Reasoning and Acting in Language Models // ICLR. — 2023.\n"
    "6. Model Context Protocol (MCP) Open Specification. — Anthropic / Linux Foundation, 2025.\n"
    "7. W3C Shapes Constraint Language (SHACL) Recommendation. — W3C, 2017.\n"
    "8. Конспекти лекцій 1-16 курсу «Сховища знань та управління знаннями», ЛНУ ім. Івана Франка, 2026."
])

# Сторінка 8: Додаток — Таблиця тверджень (Топ-20)
pdf.add_page()
add_section("Додаток А. Реєстр верифікованих тверджень проєкту (Топ-20)", [
    "Нижче наведено зведену таблицю ключових тверджень (фактів, правил і гіпотез) за результатами виконання ЛР 11-16:"
])
pdf.set_font("MainFont", "", 8.2)
claims_rows = [
    ("CL-01", "Факт", "BNPL-розстрочка регулюється стандартом PCI DSS та має ліміт 75000 грн (DOC-18)", "V2", "verified"),
    ("CL-02", "Факт", "Кешбек-картка надає 5.0% повернення для категорії MCC 5411 (DOC-06)", "V2", "verified"),
    ("CL-03", "Факт", "Алгоритм Hybrid RRF оптимізує метрику Recall@5 у каталозі e-commerce (DOC-01)", "V2", "verified"),
    ("CL-04", "Факт", "Margin-aware ранжування враховує поріг Refund Rate 25.0% (DOC-10)", "V2", "verified"),
    ("CL-05", "Правило", "Усі сутності графа повинні мати rdfs:label та prov:wasDerivedFrom (SHACL)", "V2", "verified"),
    ("CL-06", "Гіпотеза", "Дедуплікація сутностей піднімає Uniqueness із 77.8% до 100.0% (ЛР 14)", "V4", "verified"),
    ("CL-07", "Гіпотеза", "Очищення графа підвищує Faithfulness RAG-відповідей із 0.812 до 0.967", "V4", "verified"),
    ("CL-08", "Гіпотеза", "Видалення вузлів без provenance піднімає Abstention Accuracy до 1.000", "V4", "verified"),
    ("CL-09", "Правило", "Історичні тарифи зберігаються через hub:hasAlternative без втрати даних", "V3", "verified"),
    ("CL-10", "Правило", "MCP-інструменти агента працюють у суворому режимі read-only (mode=ro)", "V2", "verified"),
    ("CL-11", "Гіпотеза", "Вузол verify_citations усуває відповіді без підтвердженого doc_id", "V4", "verified"),
    ("CL-12", "Правило", "Ліміт max_steps=5 та санітизація блокують 100% атак Red Teaming (6/6)", "V3", "verified"),
    ("CL-13", "Правило", "Зміна кредитних лімітів вимагає підтвердження людини через propose_fact", "V3", "verified"),
    ("CL-14", "Гіпотеза", "Hybrid RRF (0.65/0.35) перевершує Popularity Baseline за Recall@5 та nDCG@10", "V4", "verified"),
    ("CL-15", "Правило", "Автоперевірка пояснень гарантує 100% відповідність вузлам шляху графа", "V2", "verified"),
    ("CL-16", "Правило", "Дозволити LLM прямий запис у граф без SHACL та HITL (чернетка LLM)", "V2", "rejected"),
    ("CL-17", "Правило", "Використовувати випадкове розбиття замість часового train/test (чернетка LLM)", "V3", "rejected"),
    ("CL-18", "Факт", "Включати неперевірені продукти (verified=False) у топ-5 рекомендацій", "V2", "rejected"),
    ("CL-19", "Правило", "Видаляти застарілі ціни без збереження альтернатив (чернетка LLM)", "V3", "rejected"),
    ("CL-20", "Правило", "Передавати персональні дані клієнтів (PII) у зовнішній LLM API", "V2", "rejected"),
]
for cid, ctype, ctext, cver, cstat in claims_rows:
    pdf.cell(14, 5.2, cid, border=1)
    pdf.cell(16, 5.2, ctype, border=1)
    pdf.cell(128, 5.2, ctext[:82], border=1)
    pdf.cell(12, 5.2, cver, border=1, align="C")
    pdf.cell(20, 5.2, cstat, border=1, align="C", new_x="LMARGIN", new_y="NEXT")

pdf.output("docs/whitepaper.pdf")
print("Згенеровано docs/whitepaper.pdf (кількість сторінок:", pdf.page_no(), ")")
