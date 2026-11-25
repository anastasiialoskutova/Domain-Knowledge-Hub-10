import networkx as nx

class DomainKnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        nodes = [
            ("NODE:RFM", "RFM-сегментація", ["rfm", "давність", "частота", "середній чек", "vip"], "DOC-01"),
            ("NODE:LTV", "Lifetime Value (LTV)", ["ltv", "життєва цінність", "прибуток клієнта", "arpu"], "DOC-09"),
            ("NODE:BNPL", "BNPL-розстрочка", ["bnpl", "оплата частинами", "розстрочк", "платоспроможн"], "DOC-05"),
            ("NODE:CREDIT_LIMIT", "Кредитний ліміт", ["кредитний ліміт", "ліміт", "кредитн"], "DOC-19"),
            ("NODE:ANTIFRAUD", "Антифрод-фільтр", ["антифрод", "аномальн", "сплеск", "шахрай"], "DOC-13"),
            ("NODE:MCC", "MCC-коди транзакцій", ["mcc", "кешбек", "категорія магазину", "термінал"], "DOC-10"),
            ("NODE:PCI_DSS", "Стандарт PCI DSS", ["pci dss", "токенізац", "банківськ", "захист даних"], "DOC-11"),
            ("NODE:COLD_START", "Холодний старт", ["холодн", "cold start", "новий клієнт", "без історії"], "DOC-04"),
            ("NODE:ONTOLOGY", "Онтологія SKU", ["онтологі", "атрибут", "каталог"], "DOC-20"),
            ("NODE:PAGERANK", "Персоналізований PageRank", ["pagerank", "дводольний граф", "граф"], "DOC-08"),
            ("NODE:TIME_DECAY", "Часове згасання (Time Decay)", ["time decay", "згасання", "exp(-lambda", "старих покупок"], "DOC-12"),
            ("NODE:TWO_STAGE", "Двохетапна архітектура", ["двохетапн", "retrieval", "ranking"], "DOC-14"),
            ("NODE:USER2VEC", "Ембедінги User2Vec", ["user2vec", "послідовність чеків"], "DOC-15"),
            ("NODE:SASREC", "Сесійна модель SASRec", ["sasrec", "сесійн", "трансформер", "наступний клік"], "DOC-21"),
            ("NODE:CROSS_SELL", "Крос-продажі (Cross-sell)", ["cross-sell", "крос-продаж", "апсейл", "чекаут"], "DOC-17"),
            ("NODE:MARGIN", "Маржинальність (Margin-aware)", ["маржинальн", "margin", "прибутков"], "DOC-22"),
            ("NODE:REFUND", "Штраф за повернення (Refund rate)", ["повернен", "refund", "зворотн"], "DOC-23"),
        ]
        for nid, label, synonyms, doc_id in nodes:
            self.g.add_node(nid, label=label, synonyms=synonyms, doc_id=doc_id)

        edges = [
            ("NODE:RFM", "NODE:LTV", "ВИЗНАЧАЄ_БАЗУ_ДЛЯ"),
            ("NODE:LTV", "NODE:MARGIN", "ОПТИМІЗУЄ_ЧЕРЕЗ"),
            ("NODE:BNPL", "NODE:CREDIT_LIMIT", "ПЕРЕВІРЯЄ_ОБМЕЖЕННЯ"),
            ("NODE:BNPL", "NODE:ANTIFRAUD", "ФІЛЬТРУЄТЬСЯ_ЧЕРЕЗ"),
            ("NODE:MCC", "NODE:PCI_DSS", "ЗАХИЩАЄТЬСЯ_ЗА_СТАНДАРТОМ"),
            ("NODE:MCC", "NODE:RFM", "ЗБАГАЧУЄ_ПРОФІЛЬ"),
            ("NODE:COLD_START", "NODE:ONTOLOGY", "ДОЛАЄТЬСЯ_ЧЕРЕЗ"),
            ("NODE:PAGERANK", "NODE:TIME_DECAY", "ЗВАЖУЄ_РЕБРА_ЗА"),
            ("NODE:TWO_STAGE", "NODE:USER2VEC", "ГЕНЕРУЄ_КАНДИДАТІВ_ЧЕРЕЗ"),
            ("NODE:TWO_STAGE", "NODE:SASREC", "РАНЖУЄ_СЕСІЮ_ЧЕРЕЗ"),
            ("NODE:CROSS_SELL", "NODE:MARGIN", "ВРАХОВУЄ_МАРЖУ"),
            ("NODE:CROSS_SELL", "NODE:REFUND", "ШТРАФУЄТЬСЯ_ЗА"),
        ]
        for u, v, rel in edges:
            self.g.add_edge(u, v, relation=rel)

    def entity_linking(self, query: str):
        q_low = query.lower()
        return [nid for nid, data in self.g.nodes(data=True) if any(syn in q_low for syn in data["synonyms"])]

    def extract_subgraph(self, query: str, hops: int = 2, max_edges: int = 6):
        seeds = self.entity_linking(query)
        if not seeds:
            return [], ""

        visited_nodes = set(seeds)
        collected_edges = []
        frontier = list(seeds)

        for _ in range(hops):
            next_frontier = []
            for u in frontier:
                for _, v, edata in self.g.out_edges(u, data=True):
                    edge_tuple = (u, edata["relation"], v)
                    if edge_tuple not in collected_edges and len(collected_edges) < max_edges:
                        collected_edges.append(edge_tuple)
                        visited_nodes.add(v)
                        next_frontier.append(v)
                for prev, _, edata in self.g.in_edges(u, data=True):
                    edge_tuple = (prev, edata["relation"], u)
                    if edge_tuple not in collected_edges and len(collected_edges) < max_edges:
                        collected_edges.append(edge_tuple)
                        visited_nodes.add(prev)
                        next_frontier.append(prev)
            frontier = next_frontier

        lines = []
        for nid in sorted(visited_nodes):
            ndata = self.g.nodes[nid]
            lines.append(f"[{nid}] {ndata['label']} (джерело: {ndata['doc_id']})")
        for u, rel, v in collected_edges:
            u_lbl = self.g.nodes[u]["label"]
            v_lbl = self.g.nodes[v]["label"]
            lines.append(f"[EDGE:{u}->{v}] ({u_lbl}) --[{rel}]--> ({v_lbl})")

        return sorted(visited_nodes), "\n".join(lines)

if __name__ == "__main__":
    kg = DomainKnowledgeGraph()
    nodes, text = kg.extract_subgraph("Як поєднати RFM-сегментацію та LTV для перевірки BNPL?")
    print("Знайдені вузли:", nodes)
    print("Лінеаризований підграф:\n" + text)
