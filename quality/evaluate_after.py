import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

kg_comp = pd.DataFrame([
    {"Вимір якості": "Повнота (Completeness)", "Baseline": "94.4%", "Після очищення": "100.0%", "Зміна": "+5.6%"},
    {"Вимір якості": "Походження (Provenance)", "Baseline": "83.3%", "Після очищення": "100.0%", "Зміна": "+16.7%"},
    {"Вимір якості": "Унікальність (Uniqueness)", "Baseline": "77.8%", "Після очищення": "100.0%", "Зміна": "+22.2%"},
    {"Вимір якості": "Узгодженість (Consistency)", "Baseline": "83.3%", "Після очищення": "100.0%", "Зміна": "+16.7%"},
    {"Вимір якості": "Точність (Accuracy, n=32)", "Baseline": "87.5%", "Після очищення": "100.0%", "Зміна": "+12.5%"},
    {"Вимір якості": "Актуальність (Timeliness)", "Baseline": "88.9%", "Після очищення": "100.0%", "Зміна": "+11.1%"},
    {"Вимір якості": "Валідність (SHACL Validity)", "Baseline": "83.3%", "Після очищення": "100.0%", "Зміна": "+16.7%"},
])

rag_comp = pd.DataFrame([
    {"Метрика RAG (ЛР 11, n=30)": "Faithfulness (Вірність контексту)", "Baseline KG": 0.812, "Clean KG": 0.967, "Приріст": "+0.155"},
    {"Метрика RAG (ЛР 11, n=30)": "Context Recall (Повнота пошуку)", "Baseline KG": 0.767, "Clean KG": 0.933, "Приріст": "+0.166"},
    {"Метрика RAG (ЛР 11, n=30)": "Answer Correctness (Точність)", "Baseline KG": 0.785, "Clean KG": 0.950, "Приріст": "+0.165"},
    {"Метрика RAG (ЛР 11, n=30)": "Abstention Accuracy (Відмови)", "Baseline KG": 0.750, "Clean KG": 1.000, "Приріст": "+0.250"},
])

print("=== Таблиця 5.1: Якість сховища знань ДО та ПІСЛЯ очищення ===")
print(kg_comp.to_markdown(index=False))
print("\n=== Таблиця 5.2: Вплив очищення KG на метрики RAG (ЛР 11) ===")
print(rag_comp.to_markdown(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
fig.suptitle("Вплив очищення сховища знань на якість KG та RAG (ЛР 14)", fontsize=12, fontweight="bold")

dims = ["Completeness", "Provenance", "Uniqueness", "Consistency", "Timeliness", "SHACL"]
b_vals = [94.4, 83.3, 77.8, 83.3, 88.9, 83.3]
a_vals = [100.0]*6
x1 = np.arange(len(dims))
ax1.bar(x1 - 0.18, b_vals, 0.35, label="Baseline", color="#95a5a6")
ax1.bar(x1 + 0.18, a_vals, 0.35, label="Після очищення", color="#27ae60")
ax1.set_xticks(x1); ax1.set_xticklabels(dims, rotation=20); ax1.set_ylim(60, 108)
ax1.set_ylabel("%"); ax1.set_title("1. Метрики якості сховища знань"); ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.5)

rm = ["Faithfulness", "Context Recall", "Correctness", "Abstention"]
rb = [0.812, 0.767, 0.785, 0.750]
ra = [0.967, 0.933, 0.950, 1.000]
x2 = np.arange(len(rm))
b1 = ax2.bar(x2 - 0.18, rb, 0.35, label="Baseline KG", color="#e67e22")
b2 = ax2.bar(x2 + 0.18, ra, 0.35, label="Clean KG", color="#2980b9")
ax2.set_xticks(x2); ax2.set_xticklabels(rm, rotation=15); ax2.set_ylim(0.5, 1.12)
ax2.set_title("2. Метрики якості RAG-відповідей"); ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.bar_label(b1, fmt="%.2f", padding=2, fontsize=8); ax2.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)

plt.tight_layout()
plt.savefig("figures/quality_rag_impact.png", dpi=200)
print("\nГрафік збережено: figures/quality_rag_impact.png")
