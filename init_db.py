import sqlite3
conn = sqlite3.connect("data/olap.db")
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS sales_metrics")
c.execute("CREATE TABLE sales_metrics (product TEXT, segment TEXT, quarter TEXT, conversion_pct REAL, avg_check_uah REAL)")
rows = [
    ("BNPL-розстрочка", "VIP-клієнти", "2026-Q3", 14.8, 12400.0),
    ("BNPL-розстрочка", "Loyal-клієнти", "2026-Q3", 9.6, 7800.0),
    ("Кешбек-картка", "VIP-клієнти", "2026-Q3", 18.2, 4200.0),
    ("Кредитний ліміт", "Prime-позичальники", "2026-Q3", 11.5, 19500.0),
]
c.executemany("INSERT INTO sales_metrics VALUES (?,?,?,?,?)", rows)
conn.commit()
conn.close()
print("Базу data/olap.db створено!")
