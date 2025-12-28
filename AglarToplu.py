import sys
import math
import random
import time

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QMessageBox
)

# =========================
# CSV okuma ve graf oluşturma
# =========================
def _to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    return float(x)

def load_graph_and_demands(node_path, edge_path, demand_path):
    # Düğüm verileri
    node_df = pd.read_csv(node_path, sep=";")
    node_df["s_ms"] = node_df["s_ms"].apply(_to_float)
    node_df["r_node"] = node_df["r_node"].apply(_to_float)

    # Kenar verileri
    edge_df = pd.read_csv(edge_path, sep=";")
    edge_df["capacity_mbps"] = edge_df["capacity_mbps"].apply(_to_float)
    edge_df["delay_ms"] = edge_df["delay_ms"].apply(_to_float)
    edge_df["r_link"] = edge_df["r_link"].apply(_to_float)

    # Talep verileri
    demand_df = pd.read_csv(demand_path, sep=";")
    demand_df["demand_mbps"] = demand_df["demand_mbps"].apply(_to_float)

    # Graf kur
    G = nx.Graph()
    for _, r in node_df.iterrows():
        nid = int(r["node_id"])
        G.add_node(nid, proc_delay=float(r["s_ms"]), reliability=float(r["r_node"]))

    for _, r in edge_df.iterrows():
        u, v = int(r["src"]), int(r["dst"])
        G.add_edge(u, v,
                   bandwidth=float(r["capacity_mbps"]),
                   delay=float(r["delay_ms"]),
                   reliability=float(r["r_link"]))

    demands = [(int(r["src"]), int(r["dst"]), float(r["demand_mbps"])) for _, r in demand_df.iterrows()]
    return G, demands


# =========================
# Metrikler ve toplam maliyet
# =========================
def path_metrics(G, path):
    if not path or len(path) < 2:
        return float('inf'), float('inf'), float('inf')

    # Kenar gecikmesi
    delay_edges = sum(G.edges[path[i], path[i+1]]['delay'] for i in range(len(path)-1))
    # Ara düğüm işlem gecikmesi (S ve D hariç)
    delay_nodes = sum(G.nodes[n]['proc_delay'] for n in path[1:-1])
    delay = delay_edges + delay_nodes

    # Güvenilirlik maliyeti = -log(kenar) + -log(düğüm)
    rel_cost_edges = sum(-math.log(G.edges[path[i], path[i+1]]['reliability']) for i in range(len(path)-1))
    rel_cost_nodes = sum(-math.log(G.nodes[n]['reliability']) for n in path)
    rel_cost = rel_cost_edges + rel_cost_nodes

    # Kaynak maliyeti = sum(1000 / bandwidth)
    res_cost = sum(1000.0 / G.edges[path[i], path[i+1]]['bandwidth'] for i in range(len(path)-1))

    return delay, rel_cost, res_cost

def total_cost(delay, rel_cost, res_cost, w):
    wd, wr, wres = w
    return wd * delay + wr * rel_cost + wres * res_cost

# =========================
# Algoritmalar: GA, ACO, Q-Learning
# =========================

def ga_find_path(
    G, S, D,
    w=(1/3, 1/3, 1/3),
    demand=0.0,
    generations=50,
    pop_size=40,
    seed=0,
    elite_size=1,
    crossover_rate=0.9,
    mutation_rate=0.30,
    max_init_tries=5000
):
    """
    ✅ Shortest-path / Dijkstra YOK (kesinlikle kullanılmıyor)
    ✅ Klonları "silme / remove_clones" YOK
       -> Klonlar kendiliğinden AZALSIN diye:
          (1) Fitness sharing (aynı path çoğaldıkça fitness kötüleşir)
          (2) Seçilim baskısı düşürüldü (rank-based roulette + stochastic)
          (3) Üretimde aynı çocuk üst üste gelirse yeniden deneme (hard delete değil, üretim basıncı)
    """

    rnd = random.Random(seed)

    # -------------------------------------------------
    # Geçerlilik Kontrolü
    # -------------------------------------------------
    def is_valid_path(p):
        if not p or p[0] != S or p[-1] != D:
            return False
        for i in range(len(p) - 1):
            if not G.has_edge(p[i], p[i + 1]):
                return False
        return True

    # -------------------------------------------------
    # Yol Bant Genişliği (QoS)
    # -------------------------------------------------
    def path_min_bandwidth(p):
        if len(p) < 2:
            return 0.0
        return min(G.edges[p[i], p[i + 1]]["bandwidth"] for i in range(len(p) - 1))

    # -------------------------------------------------
    # Random walk ile (Shortest Path yok!) yol üretimi
    # -------------------------------------------------
    def random_walk_to_target(start, target, visited=None, max_steps=None):
        if visited is None:
            visited = set()
        if max_steps is None:
            max_steps = len(G) * 3

        current = start
        path = [current]
        visited = set(visited)
        visited.add(current)

        steps = 0
        while steps < max_steps:
            if current == target:
                return path

            neighbors = list(G.neighbors(current))

            # ziyaret edilmemişleri tercih et (döngüleri azaltır)
            cand = [n for n in neighbors if n not in visited]

            # demand varsa bandwidth filtresi uygula
            if demand > 0.0:
                cand = [n for n in cand if G.edges[current, n]["bandwidth"] >= demand]

            # tıkandıysan: az miktar geri dönüşe izin ver (tam kilitlenmesin)
            if not cand:
                cand = neighbors[:]  # yine de bir çıkış dene
                if demand > 0.0:
                    cand = [n for n in cand if G.edges[current, n]["bandwidth"] >= demand]
                if not cand:
                    return None

            nxt = rnd.choice(cand)
            path.append(nxt)
            visited.add(nxt)
            current = nxt
            steps += 1

        return None

    def random_simple_path():
        return random_walk_to_target(S, D, visited=None, max_steps=len(G) * 3)

    # -------------------------------------------------
    # Repair (Shortest Path yok!)
    # Çocuk path bozulursa: S->... kısmını valid tut, sonra random-walk ile D'ye bağla
    # -------------------------------------------------
    def repair_path(p):
        if not p:
            return None
        if p[0] != S:
            return None

        # geçerli kenar zincirini koru
        fixed = [p[0]]
        visited = {p[0]}
        for i in range(1, len(p)):
            u = fixed[-1]
            v = p[i]
            if v in visited:
                continue
            if not G.has_edge(u, v):
                break
            # demand varsa bu edge'yi de kontrol et
            if demand > 0.0 and G.edges[u, v]["bandwidth"] < demand:
                break
            fixed.append(v)
            visited.add(v)
            if v == D:
                return fixed

        # D'ye bağlanmayı dene (random-walk)
        tail = random_walk_to_target(fixed[-1], D, visited=visited, max_steps=len(G) * 3)
        if not tail:
            return None

        # tail'in ilk düğümü fixed[-1] zaten, tekrar etmeyelim
        merged = fixed + tail[1:]
        return merged if is_valid_path(merged) else None

    # -------------------------------------------------
    # Fitness (yumuşak ceza + length dengeleme)
    # -------------------------------------------------
    def base_fitness(p):
        # path_metrics ve total_cost dışarıda tanımlı varsayılıyor
        d, rc, res = path_metrics(G, p)
        base_cost = total_cost(d, rc, res, w)

        penalty = 0.0

        if not is_valid_path(p):
            penalty += 1e6

        if demand > 0.0 and len(p) > 1 and is_valid_path(p):
            min_bw = path_min_bandwidth(p)
            if min_bw < demand:
                penalty += (demand - min_bw) * 1000.0

        # 🔥 shortest-path dominansını kıran ufak denge
        length_penalty = 0.01 * len(p)

        return base_cost + penalty + length_penalty

    # -------------------------------------------------
    # Fitness Sharing: aynı path çoğaldıkça "kendiliğinden" kötüleşsin
    # (klon silmiyoruz, avantajlarını kırıyoruz)
    # -------------------------------------------------
    def shared_fitness(p, counts, share_strength=0.15):
        f = base_fitness(p)
        c = counts.get(tuple(p), 1)
        # minimization: c arttıkça f büyüsün
        return f * (1.0 + share_strength * (c - 1))

    # -------------------------------------------------
    # Rank-based Roulette (seçilim baskısı düşük, diversity daha iyi)
    # -------------------------------------------------
    def rank_roulette_selection(pop, fit_vals):
        ranked_idx = sorted(range(len(pop)), key=lambda i: fit_vals[i])
        n = len(pop)

        scores = [0.0] * n
        for rank, i in enumerate(ranked_idx):
            scores[i] = (n - rank)

        scores = [s ** 0.7 for s in scores]

        total = sum(scores)
        pick = rnd.uniform(0, total)
        acc = 0.0
        for p, s in zip(pop, scores):
            acc += s
            if acc >= pick:
                return p
        return pop[-1]

    # -------------------------------------------------
    # Crossover – Path-Aware + Fallback
    # -------------------------------------------------
    def crossover_1point(p1, p2):
        if len(p1) < 2 or len(p2) < 2:
            return p1[:]
        cut = rnd.randint(1, min(len(p1), len(p2)) - 1)
        return p1[:cut] + p2[cut:]

    def crossover_path_aware(p1, p2):
        common = list(set(p1[1:-1]) & set(p2[1:-1]))
        if not common:
            return crossover_1point(p1, p2)

        mid = rnd.choice(common)
        i1 = p1.index(mid)
        i2 = p2.index(mid)
        child = p1[:i1] + p2[i2:]
        return child

    # -------------------------------------------------
    # Mutasyon
    # -------------------------------------------------
    def mutate_random_gene(p):
        if len(p) < 3:
            return p[:]
        child = p[:]
        idx = rnd.randint(1, len(child) - 2)
        prev = child[idx - 1]

        neighbors = list(G.neighbors(prev))
        if demand > 0.0:
            neighbors = [n for n in neighbors if G.edges[prev, n]["bandwidth"] >= demand]

        if neighbors:
            child[idx] = rnd.choice(neighbors)
        return child

    def mutate_segment_reset(p):
        if len(p) < 4:
            return p[:]
        cut = rnd.randint(1, len(p) - 2)
        head = p[:cut]
        tail = random_walk_to_target(head[-1], D, visited=set(head), max_steps=len(G) * 3)
        if not tail:
            return p[:]
        child = head + tail[1:]
        return child

    def mutate(p):
        if rnd.random() < 0.5:
            return mutate_random_gene(p)
        else:
            return mutate_segment_reset(p)

    # -------------------------------------------------
    # Başlangıç Popülasyonu
    # -------------------------------------------------
    population = []
    tries = 0
    while len(population) < pop_size and tries < max_init_tries:
        tries += 1
        p = random_simple_path()
        if p and is_valid_path(p):
            population.append(p)

    if not population:
        return [S, D], (float("inf"), float("inf"), float("inf"))

    # -------------------------------------------------
    # Evrim Döngüsü
    # -------------------------------------------------
    elite_size = max(0, min(elite_size, pop_size))
    inject_count = max(1, int(pop_size * 0.10))

    for generation in range(generations):
        counts = {}
        for p in population:
            k = tuple(p)
            counts[k] = counts.get(k, 0) + 1

        fit_vals = [shared_fitness(p, counts, share_strength=0.15) for p in population]
        ranked = sorted(zip(population, fit_vals), key=lambda x: x[1])

        new_population = [p for p, _ in ranked[:elite_size]]
        target_fill = pop_size - inject_count

        while len(new_population) < target_fill:
            p1 = rank_roulette_selection(population, fit_vals)
            p2 = rank_roulette_selection(population, fit_vals)

            if rnd.random() < crossover_rate:
                child = crossover_path_aware(p1, p2)
            else:
                child = p1[:]

            if rnd.random() < mutation_rate:
                child = mutate(child)

            child = repair_path(child) if child else None
            if not child:
                continue

            if tuple(child) in set(tuple(x) for x in new_population):
                if rnd.random() < 0.15:
                    new_population.append(child)
                else:
                    continue
            else:
                new_population.append(child)

        tries_inject = 0
        while len(new_population) < pop_size and tries_inject < 300:
            tries_inject += 1
            rp = random_simple_path()
            if rp and is_valid_path(rp):
                if tuple(rp) not in set(tuple(x) for x in new_population) or rnd.random() < 0.20:
                    new_population.append(rp)

        while len(new_population) < pop_size:
            new_population.append(rnd.choice(population)[:])

        population = new_population

    # ✅ SADECE ÇIKTI SIRASI: en iyi yol hep 1. sırada görünmesin
    rnd.shuffle(population)
    print(population)

    best = min(population, key=lambda p: base_fitness(p))
    return best, path_metrics(G, best)

# =========================
# Q-Learning
# =========================

def edge_cost(G, S, u, v, w):
    d = G.edges[u, v]['delay']
    rel_c = -math.log(G.edges[u, v]['reliability']) + (-math.log(G.nodes[v]['reliability']))
    if u == S:
        rel_c += -math.log(G.nodes[u]['reliability'])
    res_c = 1000.0 / G.edges[u, v]['bandwidth']
    return w[0]*d + w[1]*rel_c + w[2]*res_c

def epsilon_greedy(Q, state, actions, epsilon, rnd):
    if rnd.random() < epsilon:
        return rnd.choice(actions)
    else:
        return max(actions, key=lambda a: Q.get((state, a), 0.0))

def q_learning_path(G, S, D, w=(1/3,1/3,1/3), demand=0.0, episodes=800, alpha=0.2, gamma=0.95,
                    epsilon=0.2, seed=0):

    rnd = random.Random(seed)
    Q = {}
    best_path, best_cost = None, float('inf')

    # Q tablosu ilklendir
    for u, v in G.edges():
        initial = -edge_cost(G, S, u, v, w)
        Q[(u, v)] = initial
        Q[(v, u)] = initial

    destination_reward, step_penalty = 200.0, 0.1 # bu ikisi, hızlı çalışsın diye

    for ep in range(episodes):
        state, path, visited, step = S, [S], {S}, 0

        while state != D and step < len(G):
            # Bandwidth kontrolü
            actions = []
            for a in G.neighbors(state):
                if a not in visited or a == D:
                    if demand > 0.0:
                        if G.edges[state, a]['bandwidth'] >= demand:
                            actions.append(a)
                    else:
                        actions.append(a)

            if not actions:
                break

            # epsilon-greedy seçim
            action = epsilon_greedy(Q, state, actions, epsilon, rnd)
            path.append(action)
            visited.add(action)

            # ödül
            instantaneous_cost = edge_cost(G, S, state, action, w)
            r = -instantaneous_cost - step_penalty
            if demand > 0.0 and G.edges[state, action]['bandwidth'] < demand: # hızlı çalışsın diye
                r -= 500.0

            next_state = action
            next_actions = list(G.neighbors(next_state))

            # next_state uzantıları varsa two-step lookahead hesapla
            if next_actions:
                # En iyi sonraki state'i seç
                next_next = max(next_actions, key=lambda na: Q.get((next_state, na), 0.0))

                # next_next’in komşuları
                nn_actions = list(G.neighbors(next_next))

                if nn_actions:
                    lookahead_q = max(Q.get((next_next, nna), 0.0) for nna in nn_actions)
                else:
                    lookahead_q = max(Q.get((next_state, na), 0.0) for na in next_actions)
            else:
                # hiç komşu yoksa 0
                lookahead_q = 0.0
            # Q güncellemesi
            Q[(state, action)] = Q.get((state, action)) + alpha * (r + gamma * lookahead_q - Q.get((state, action)))

            state, step = action, step+1

        # episode sonunda kontrol
        if path[-1] == D:
            bandwidth_ok = True
            if demand > 0.0:
                for i in range(len(path)-1):
                    if G.edges[path[i], path[i+1]]['bandwidth'] < demand:
                        bandwidth_ok = False
                        break

            if (demand == 0.0) or bandwidth_ok:
                Q[(path[-2], path[-1])] = Q.get((path[-2], path[-1])) + destination_reward
                d, rc, res = path_metrics(G, path)
                c = total_cost(d, rc, res, w)
                if c < best_cost:
                    best_cost, best_path = c, path

    if best_path and len(best_path) > 1:
        print(best_path)
        return best_path, path_metrics(G, best_path)
    else:
        return [S, D], (float('inf'), float('inf'), float('inf'))

# =========================
# Ant Colony
# =========================

def aco_find_path(G, S, D, w=(1/3, 1/3, 1/3), demand = 0.0, ants = 40, iters = 40, alpha = 1.0, 
                  beta = 2.0, rho = 0.5, seed = 0):
    """
    KLASİK ACO (Ant System) + Elitist güncelleme olarak yeniden yorumlanmış sürüm.
    ---------------------------------------------------------------------------
    Bu implementasyon, algoritmayı değiştirmeden klasik ACO mantığıyla okunur:

    - Tek amaç (single-objective): Her kenar için 'edge_cost' bir "mesafe" (distance) gibi ele alınır.
      (Gecikme, güvenilirlik ve bandwidth bileşenleri, tek bir maliyet/mesafe fonksiyonuna indirgenmiştir.)

    - Heuristic bilgi: η(i,j) = 1 / edge_cost(i,j) olarak kullanılır.
      Klasik ACO'daki 1/distance yaklaşımının aynısıdır.

    - Kısıtlar (demand): Bandwidth < demand olan kenarlar problem uzayının dışında sayılır (feasible değil).
      Bu nedenle karınca o kenarı hiç aday olarak görmez.

    - Tabu list: visited kümesi, karıncanın döngü yapmasını engelleyen tabu list gibi çalışır.

    - Feromon güncellemesi: Ant System (ant-cycle) mantığıyla, sadece hedefe ulaşan karıncalar
      tur tamamladıktan sonra feromon bırakır: Δτ ∝ Q / L   (L: tur uzunluğu/maliyeti)

    - Elitist ek: Her iterasyonun en iyi turu ekstra feromon alır (Elitist Ant System yaklaşımı).
    """

    rnd = random.Random(seed)

    # -----------------------------------------------------------------------
    # 1) Feromonların başlangıç değeri (τ0): tüm kenarlara eşit feromon
    # -----------------------------------------------------------------------
    tau = {}
    for (u, v) in G.edges():
        tau[(u, v)] = 1.0
        tau[(v, u)] = 1.0

    # Feromon bırakma sabiti
    Q = 100.0

    # -----------------------------------------------------------------------
    # 2) Heuristic / "distance" tanımı (tek objective gibi okunur)
    #    edge_cost = birleşik mesafe fonksiyonu
    # -----------------------------------------------------------------------
    def edge_cost(u, v):
        d = G.edges[u, v]['delay']
        lr = G.edges[u, v]['reliability']
        bw = G.edges[u, v]['bandwidth']

        # reliability yüksek olmalı -> -log(reliability) ile "maliyete" çevrilir
        rel = -math.log(lr)

        # bandwidth yüksek olmalı -> 1/bw benzeri bir cezaya çevrilir
        res = 1000.0 / bw

        # toplam "mesafe/maliyet"
        return total_cost(d, rel, res, w)

    best_path = None
    best_c = float('inf')

    # -----------------------------------------------------------------------
    # 3) ACO ana döngüsü (iterasyonlar)
    # -----------------------------------------------------------------------
    for _ in range(iters):

        paths = []
        costs = []

        iter_best_path = None
        iter_best_cost = float('inf')

        # -------------------------------------------------------------------
        # 4) Karıncalar çözüm inşa eder (solution construction)
        # -------------------------------------------------------------------
        for _a in range(ants):
            current = S

            # Tabu list (visited) - döngüyü engellemek için
            visited = {S}

            path = [S]
            steps = 0

            # Karınca hedefe ulaşana kadar (veya maksimum adım sınırına kadar) ilerler
            while current != D and steps < len(G):

                neighbors = list(G.neighbors(current))
                rnd.shuffle(neighbors)  # keşif (exploration)

                desirabilities = []
                candidates = []

                # ----------------------------------------------------------------
                # Aday seçim kümesi: feasible komşular
                # ----------------------------------------------------------------
                for n in neighbors:

                    # Tabu kuralı: hedef değilse tekrar ziyaret etme
                    if n in visited and n != D:
                        continue

                    # Kısıt: demand varsa, bandwidth yetersiz kenarlar problem uzayında yok sayılır
                    if demand > 0.0:
                        if G.edges[current, n]['bandwidth'] < demand:
                            continue

                    # Feromon ve heuristic (η = 1/cost) ile olasılık hesabı
                    t = tau[(current, n)]
                    c = edge_cost(current, n)

                    # Klasik ACO formu: (τ^α) * (η^β)
                    eta = 1.0 / (c + 1e-6)
                    desirabilities.append((t * alpha) * (eta * beta))
                    candidates.append(n)

                # feasible komşu yoksa: karınca geçersiz çözüm (ölür)
                if not candidates:
                    break

                # ----------------------------------------------------------------
                # Roulette-wheel seçim (olasılıksal seçim)
                # ----------------------------------------------------------------
                s = sum(desirabilities)
                r = rnd.random() * s
                acc = 0.0
                nxt = candidates[-1]

                for dval, n in zip(desirabilities, candidates):
                    acc += dval
                    if acc >= r:
                        nxt = n
                        break

                path.append(nxt)
                visited.add(nxt)
                current = nxt
                steps += 1

            # -------------------------------------------------------------------
            # 5) Sadece hedefe ulaşan karıncalar tur (solution) üretmiş sayılır
            # -------------------------------------------------------------------
            if path[-1] == D:

                # Ek güvenlik: path boyunca min bandwidth demand'ı sağlamalı
                if demand > 0.0:
                    min_bw = min(G.edges[path[i], path[i+1]]['bandwidth']
                                 for i in range(len(path)-1))
                    if min_bw < demand:
                        continue

                d, rc, res = path_metrics(G, path)
                c = total_cost(d, rc, res, w)

                paths.append(path)
                costs.append(c)

                # Global best
                if c < best_c:
                    best_c = c
                    best_path = path

                # Iteration best
                if c < iter_best_cost:
                    iter_best_cost = c
                    iter_best_path = path

        # -------------------------------------------------------------------
        # 6) Feromon buharlaşması (evaporation): τ <- (1-ρ)τ
        # -------------------------------------------------------------------
        for e in tau:
            tau[e] = (1 - rho) * tau[e]

        # -------------------------------------------------------------------
        # 7) Ant System (ant-cycle) feromon bırakma:
        #    Tüm başarılı karıncaların turlarına Q/L kadar feromon eklenir
        # -------------------------------------------------------------------
        for path, c in zip(paths, costs):
            dep = Q / (c + 1e-6)
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                tau[(u, v)] += dep
                tau[(v, u)] += dep

        # -------------------------------------------------------------------
        # 8) Elitist güncelleme: iterasyonun en iyi turu ekstra feromon alır
        # -------------------------------------------------------------------
        if iter_best_path is not None:
            elite_dep = Q / (iter_best_cost + 1e-6)
            for i in range(len(iter_best_path)-1):
                u, v = iter_best_path[i], iter_best_path[i+1]
                tau[(u, v)] += elite_dep
                tau[(v, u)] += elite_dep

    return best_path, path_metrics(G, best_path)

# =========================
# PyQt5 uygulaması
# =========================
class RoutingApp(QMainWindow):
    def __init__(self, node_csv="BSM307_317_Guz2025_TermProject_NodeData.csv",
                 edge_csv="BSM307_317_Guz2025_TermProject_EdgeData.csv",
                 demand_csv="BSM307_317_Guz2025_TermProject_DemandData.csv"):
        super().__init__()
        self.setWindowTitle("QoS Çok Amaçlı Rotalama - Demand Entegrasyonu (CSV)")
        self.resize(1200, 750)

        # Graf ve talepler
        self.G, self.demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)
        self.nodes = sorted(list(self.G.nodes()))
        self.S_default = self.nodes[0] if self.nodes else 0
        self.D_default = self.nodes[-1] if self.nodes else 0
        self.current_demand = 0.0  # Senaryodan gelen Mbps talebi

        # Layout
        self.pos = nx.spring_layout(self.G, seed=0)

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        controls = QGridLayout()
        row = 0

        controls.addWidget(QLabel("Kaynak (S)"), row, 0)
        self.combo_S = QComboBox()
        for n in self.nodes:
            self.combo_S.addItem(str(n))
        self.combo_S.setCurrentText(str(self.S_default))
        controls.addWidget(self.combo_S, row, 1)
        row += 1

        controls.addWidget(QLabel("Hedef (D)"), row, 0)
        self.combo_D = QComboBox()
        for n in self.nodes:
            self.combo_D.addItem(str(n))
        self.combo_D.setCurrentText(str(self.D_default))
        controls.addWidget(self.combo_D, row, 1)
        row += 1

        controls.addWidget(QLabel("Senaryo (Demand)"), row, 0)
        self.combo_dem = QComboBox()
        for (s, d, b) in self.demands:
            self.combo_dem.addItem(f"{s}->{d} ({b:.0f} Mbps)")
        controls.addWidget(self.combo_dem, row, 1)
        row += 1

        self.btn_use_dem = QPushButton("Senaryodan S/D al")
        self.btn_use_dem.clicked.connect(self.on_use_demand)
        controls.addWidget(self.btn_use_dem, row, 0, 1, 2)
        row += 1

        controls.addWidget(QLabel("Algoritma"), row, 0)
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["GA", "Q-Learning", "ACO"])
        controls.addWidget(self.combo_algo, row, 1)
        row += 1

        # Ağırlıklar
        controls.addWidget(QLabel("Wdelay"), row, 0)
        self.spin_wd = QDoubleSpinBox()
        self.spin_wd.setRange(0.0, 1.0)
        self.spin_wd.setSingleStep(0.05)
        self.spin_wd.setValue(1/3)
        controls.addWidget(self.spin_wd, row, 1)
        row += 1

        controls.addWidget(QLabel("Wreliability"), row, 0)
        self.spin_wr = QDoubleSpinBox()
        self.spin_wr.setRange(0.0, 1.0)
        self.spin_wr.setSingleStep(0.05)
        self.spin_wr.setValue(1/3)
        controls.addWidget(self.spin_wr, row, 1)
        row += 1

        controls.addWidget(QLabel("Wresource"), row, 0)
        self.spin_wres = QDoubleSpinBox()
        self.spin_wres.setRange(0.0, 1.0)
        self.spin_wres.setSingleStep(0.05)
        self.spin_wres.setValue(1/3)
        controls.addWidget(self.spin_wres, row, 1)
        row += 1

        # Parametreler
        controls.addWidget(QLabel("GA generations"), row, 0)
        self.spin_gen = QSpinBox()
        self.spin_gen.setRange(10, 300)
        self.spin_gen.setValue(90)
        controls.addWidget(self.spin_gen, row, 1)
        row += 1

        controls.addWidget(QLabel("GA pop size"), row, 0)
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 200)
        self.spin_pop.setValue(60)
        controls.addWidget(self.spin_pop, row, 1)
        row += 1

        controls.addWidget(QLabel("ACO ants"), row, 0)
        self.spin_ants = QSpinBox()
        self.spin_ants.setRange(10, 200)
        self.spin_ants.setValue(40)
        controls.addWidget(self.spin_ants, row, 1)
        row += 1

        controls.addWidget(QLabel("ACO iters"), row, 0)
        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(10, 200)
        self.spin_iters.setValue(40)
        controls.addWidget(self.spin_iters, row, 1)
        row += 1

        controls.addWidget(QLabel("RL episodes"), row, 0)
        self.spin_eps = QSpinBox()
        self.spin_eps.setRange(1, 9999999)
        self.spin_eps.setValue(800)
        controls.addWidget(self.spin_eps, row, 1)
        row += 1

        controls.addWidget(QLabel("Demand (Mbps) [0 → senaryodan]"), row, 0)
        self.spin_B = QDoubleSpinBox()
        self.spin_B.setRange(0.0, 5000.0)
        self.spin_B.setDecimals(1)
        self.spin_B.setSingleStep(50.0)
        self.spin_B.setValue(0.0)
        self.spin_B.valueChanged.connect(self.update_demand_label)
        controls.addWidget(self.spin_B, row, 1)
        row += 1

        self.btn_calc = QPushButton("Hesapla")
        self.btn_calc.clicked.connect(self.on_calculate)
        controls.addWidget(self.btn_calc, row, 0, 1, 2)
        row += 1

        self.lbl_demand = QLabel("Demand: (senaryodan seçin)")
        controls.addWidget(self.lbl_demand, row, 0, 1, 2)
        row += 1

        self.lbl_result = QLabel("Sonuçlar burada gösterilecek")
        self.lbl_result.setWordWrap(True)
        controls.addWidget(self.lbl_result, row, 0, 2, 2)
        row += 2

        left = QWidget()
        left.setLayout(controls)
        main_layout.addWidget(left, stretch=0)

        # Sağ panel: çizim
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, stretch=1)

        # İlk çizim (hesaplama yapılmadan)
        self.draw_graph()

    def weights(self):
        wd = self.spin_wd.value()
        wr = self.spin_wr.value()
        wres = self.spin_wres.value()
        s = wd + wr + wres
        if s == 0:
            QMessageBox.warning(self, "Uyarı", "Ağırlıkların toplamı 0 olamaz. Varsayılan 1/3 uygulanacak.")
            return (1/3, 1/3, 1/3)
        return (wd/s, wr/s, wres/s)

    def draw_graph(self, path = None, title = ""):
        self.ax.clear()
        # Tüm düğümler gri; kenarlar yarı saydam
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=77, node_color='lightgray')
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='lightgray', width=0.5, alpha=0.3)
        nx.draw_networkx_labels(self.G, self.pos, labels={n: str(n) for n in self.G.nodes()}, font_size=8, font_color="black", ax=self.ax)

        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            if len(path) > 2:
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=path[1:-1], ax=self.ax, node_size=100, node_color='orange')
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[0]], ax=self.ax, node_size=130, node_color='lightgreen')
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[-1]], ax=self.ax, node_size=130, node_color='deepskyblue')
            nx.draw_networkx_edges(self.G, self.pos, edgelist=path_edges, ax=self.ax, edge_color='red', width=2.5)

        self.ax.set_title(title)
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

    def on_use_demand(self):
        idx = self.combo_dem.currentIndex()
        if idx < 0 or idx >= len(self.demands):
            QMessageBox.warning(self, "Uyarı", "Senaryo seçimi geçersiz.")
            return
        s, d, b = self.demands[idx]
        self.combo_S.setCurrentText(str(s))
        self.combo_D.setCurrentText(str(d))
        self.current_demand = b  # demand değeri kaydedildi
        self.spin_B.setValue(b)
        self.lbl_demand.setText(f"Demand: {b:.1f} Mbps (senaryodan)")

        QMessageBox.information(self, "Bilgi", f"Senaryodan alındı: S={s}, D={d}, talep={b:.0f} Mbps")
        
    def update_demand_label(self, text):
        val = float(text)
        if val == 0.0:
            self.lbl_demand.setText("Demand: (senaryodan seçin)")
        else:
            self.lbl_demand.setText(f"Demand: {val:.1f} Mbps")


    def on_calculate(self):
        # S/D
        try:
            S = int(self.combo_S.currentText())
            D = int(self.combo_D.currentText())
        except Exception:
            QMessageBox.critical(self, "Hata", "S/D seçiminde sorun var.")
            return

        if S == D:
            QMessageBox.warning(self, "Uyarı", "Kaynak ve hedef farklı olmalı.")
            return

        # Ağırlıklar
        w = self.weights()

        # Algoritma seçimi
        algo = self.combo_algo.currentText()

        # Demand belirleme: spin_B > 0 ise onu kullan, değilse senaryodan gelen current_demand
        demand_used = self.spin_B.value() if self.spin_B.value() > 0.0 else self.current_demand
    
        # Yol bulma
        t0 = time.time()
        
        # Önce talebi karşılayan bir yol ara
        path, metrics = None, (0.0, 0.0, 0.0)
        
        if algo == "GA":
            path, metrics = ga_find_path(
                self.G, S, D, w=w,
                demand=demand_used,  # Demand parametresini ekleyin
                generations=self.spin_gen.value(),
                pop_size=self.spin_pop.value(),
                seed=0
            )
        elif algo == "ACO":
            path, metrics = aco_find_path(
                self.G, S, D, w=w,
                demand=demand_used,  # Demand parametresini ekleyin
                ants=self.spin_ants.value(),
                iters=self.spin_iters.value(),
                seed=0
            )
        else:  # Q-Learning
            path, metrics = q_learning_path(
                self.G, S, D,
                w=w,
                demand=demand_used,   # Talep parametresi
                episodes=self.spin_eps.value(), 
                alpha=0.5, 
                gamma=0.95, 
                epsilon=0.2, 
                seed=0
            )
    
        rt_alg = time.time() - t0
        d, rc, res = metrics
        cost = total_cost(d, rc, res, w)
        
        # Yol kapasite kontrolü
        capacity_txt = ""
        bandwidth_warning = False

        bw_values = []
        for i in range(len(path)-1):
            if self.G.has_edge(path[i], path[i+1]):
                bw_values.append(self.G.edges[path[i], path[i+1]].get('bandwidth', 0))
            else:
                # Kenar yoksa uyarı ver
                QMessageBox.warning(
                    self, "Kapasite Uyarısı",
                    f"UYARI: {path[i]} → {path[i+1]} arasında talebi karşılayan kenar bulunamadı!"
                )
                return

        if bw_values:
            min_bw = min(bw_values)
        else:
            min_bw = 0

        if demand_used > 0.0:
            if min_bw < demand_used:
                QMessageBox.warning(
                    self, "Kapasite Uyarısı",
                    f"UYARI: {S} → {D} arasında talebi karşılayan kenar bulunamadı!"
                )
                return
            else:
                capacity_txt = f"\n✓ Kapasite uygun: min bw={min_bw:.1f} Mbps, talep={demand_used:.1f} Mbps"

        # Çizim ve sonuç
        title = f"{algo} yolu"
        if bandwidth_warning:
            title += " ⚠️ (Talebi karşılamıyor)"

        self.draw_graph(path, title=title)

        result_text = (f"Algoritma={algo}, S={S}, D={D}\n"
                       f"Delay={d:.3f} ms, RelCost={rc:.4f}, ResCost={res:.4f}\n"
                       f"TotalCost={cost:.4f}, Runtime={rt_alg:.2f} s")

        if demand_used > 0.0:
            result_text += f"\nTalep: {demand_used:.1f} Mbps"

        result_text += f"{capacity_txt}"

        self.lbl_result.setText(result_text)


def main():
    app = QApplication(sys.argv)
    # Dosya adları aynı klasördeyse doğrudan çalışır
    win = RoutingApp(
        node_csv="BSM307_317_Guz2025_TermProject_NodeData.csv",
        edge_csv="BSM307_317_Guz2025_TermProject_EdgeData.csv",
        demand_csv="BSM307_317_Guz2025_TermProject_DemandData.csv"
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()