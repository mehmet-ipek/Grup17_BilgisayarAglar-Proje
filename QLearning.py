import math
import time
import random
import numpy as np
from Metrix import path_metrics, total_cost
from Network import load_graph_and_demands

def between_2_node_cost(G, S, u, v, w):
    """
    İki düğüm arasındaki maliyeti hesaplar.
    """
    
    d = G.edges[u, v]['delay'] #gecikme
    rel_c = -math.log(G.edges[u, v]['reliability']) + (-math.log(G.nodes[v]['reliability'])) #güvenilirlik maliyeti
    if u == S: # başlangıç düğümü için güvenilirlik maliyeti
        rel_c += -math.log(G.nodes[u]['reliability'])
    res_c = 1000.0 / G.edges[u, v]['bandwidth'] # kaynak maliyeti
    return w[0]*d + w[1]*rel_c + w[2]*res_c

def epsilon_greedy(Q, state, actions, epsilon, rnd):
    """
    Epsilon-greedy stratejisi ile bir eylem seçer.
    """
    
    if rnd.random() < epsilon:
        return rnd.choice(actions) #explore (rastgele eylem seçer)
    else:
        return max(actions, key=lambda a: Q[state, a]) #exploit (Q tablosuna göre en iyi eylemi seçer)

def q_learning_path(G, S, D, w=(1/3,1/3,1/3), demand=0.0, episodes=800, alpha=0.2, gamma=0.95,
                    epsilon=0.2, seed=42):
    """
    Q-learning ile Source -> Destination arasındaki en düşük maliyetli yolu bulur.
    
        - Her adımda u->v geçiş maliyeti hesaplanır
        - Ajan epsilon-greedy ile komşulardan birini seçer
        - Q tablosu, güncelleme formülü ile güncellenir
        - Hedefe ulaşan yollara ek ödül verilir
    """

    rnd = random.Random(seed)
    best_path, best_cost = None, float('inf')
    
    n = len(G.nodes()) #Düğüm sayısı
    Q = np.zeros((n, n)) #Q tablosu
    
    # Q tablosunu ilklendir
    for u, v in G.edges():
        initial = -between_2_node_cost(G, S, u, v, w)
        Q[u, v] = initial
        Q[v, u] = initial #hem Q[u, v] = initial hem de Q[v, u] = initial olması ile çift yönlü yapıldı

    destination_reward, step_penalty = 200.0, 0.1 # hedefe ulaşma ödülü ve adım cezası

    for i in range(episodes):
        state, path, visited, step = S, [S], {S}, 0

        while state != D and step < len(G): # Çok uzun dolaşmayı engellemek için max step limiti
            # Bu durumdan gidilebilecek geçerli tüm eylemleri (komşuları) topla
            actions = []
            for a in G.neighbors(state):
                if a not in visited or a == D:
                    if demand > 0.0: # Bant genişliği talebi varsa ve talep karşılanıyorsa listeye ekle
                        if G.edges[state, a]['bandwidth'] >= demand:
                            actions.append(a)
                    else:
                        actions.append(a)

            if not actions: # hiç action yoksa sonraki episode a geç
                break

            # epsilon-greedy seçim
            action = epsilon_greedy(Q, state, actions, epsilon, rnd)
            path.append(action)
            visited.add(action)

            # Geçiş maliyeti → ödül = -(cost + step_penalty)
            instantaneous_cost = between_2_node_cost(G, S, state, action, w)
            r = -instantaneous_cost - step_penalty

            next_state = action
            # Sonraki durumun eylemlerinden bant genişliği talebini sağlayan eylemleri listeye ekle
            next_actions = list(filter(lambda a: (demand == 0.0 or G.edges[next_state, a]['bandwidth'] >= demand),
                                       G.neighbors(next_state))) 

            # gelecek durum için en iyi Q değeri max(Q(s',a'))
            if next_actions:
                max_next = max(Q[next_state, na] for na in next_actions)
            else:
                # hiç komşu yoksa 0
                max_next = 0.0
            # Q tablosu güncelleme formülü
            Q[state, action] = Q[state, action] + alpha * (r + gamma * max_next - Q[state, action])

            state, step = action, step+1

        # episode sonunda kontrol
        if path[-1] == D: # Hedefe ulaşıldı mı
            bandwidth_ok = True
            if demand > 0.0: # bandwidth talebi var mı
                for i in range(len(path)-1):
                    if G.edges[path[i], path[i+1]]['bandwidth'] < demand: # talebi karşılayan yol bulundu mu
                        bandwidth_ok = False
                        break

            if (demand == 0.0) or bandwidth_ok:
                Q[path[-2], path[-1]] = Q[path[-2], path[-1]] + destination_reward # hedefe ulaşan yollara ödül ver
                d, rc, res = path_metrics(G, path)
                c = total_cost(d, rc, res, w) # topplam maliyeti hesapla
                if c < best_cost:
                    best_cost, best_path = c, path # en düşük maliyetli yolu ve onun maliyetini tut
    
    # En iyi yol bulunduysa döndür
    if best_path and len(best_path) > 1:
        return best_path, path_metrics(G, best_path), Q
    else:# Yol bulunamadıysa source ve destination arasında sonsuz döndür
        return [S, D], (float('inf'), float('inf'), float('inf')), Q
    
if __name__ == "__main__":
    start_time = time.time() #başlangıç zamanı
    G, demands = load_graph_and_demands() # ağ ve talepleri al

    weights = (1/3,1/3,1/3)
    demand = 0.0

    best_path, metrics, Q = q_learning_path(G, 0, 249, weights, demand, 800, 0.2, 0.95, 0.2, 42)
    totalCost = total_cost(metrics[0], metrics[1], metrics[2], weights) # toplam maliyet
    
    end_time = time.time()#bitiş zamanı
    runtime = end_time - start_time # çalışma süresi
    #sonuçları yazdır
    if totalCost == float('inf'):
        print("Yol bulunamadı!")
    else:
        print(f"Best path: {best_path}")
        print(f"Delay: {metrics[0]} ms\tReliablity cost: {metrics[1]}\tResource cost: {metrics[2]}")
        print(f"Total Cost: {totalCost}")
    print(f"Runtime: {runtime:.4f} seconds")
