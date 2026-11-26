import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
fig.suptitle("Оцінювання якості та безпеки LLM-агента Domain Knowledge Hub (ЛР 15)", fontsize=11.5, fontweight="bold")

cats = ["Успішні\n(n=5)", "Не знаю\n(n=3)", "Конфлікт\n(n=2)", "Загалом\n(n=10)"]
before_sc = [60.0, 100.0, 50.0, 70.0]
after_sc = [100.0, 100.0, 100.0, 100.0]
x1 = np.arange(len(cats))
b1 = ax1.bar(x1 - 0.18, before_sc, 0.35, label="Прогін 1 (Baseline)", color="#95a5a6")
b2 = ax1.bar(x1 + 0.18, after_sc, 0.35, label="Прогін 2 (Виправлено)", color="#2980b9")
ax1.set_xticks(x1); ax1.set_xticklabels(cats); ax1.set_ylim(0, 115)
ax1.set_ylabel("Success Rate, %"); ax1.set_title("1. Сценарні тести (до / після виправлень)")
ax1.legend(); ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.bar_label(b1, fmt="%.0f%%", padding=2, fontsize=8); ax1.bar_label(b2, fmt="%.0f%%", padding=2, fontsize=8)

sec_labels = ["До захисту\n(Baseline)", "Після захисту\n(Guardrails + RO)"]
sec_vals = [16.7, 100.0]
bars = ax2.bar(sec_labels, sec_vals, width=0.45, color=["#e74c3c", "#27ae60"])
ax2.set_ylim(0, 115); ax2.set_ylabel("Заблоковано атак, %")
ax2.set_title("2. Red Teaming: блокування атак (n=6)")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/agent_eval_dashboard.png", dpi=200)
print("Графік збережено: figures/agent_eval_dashboard.png")
