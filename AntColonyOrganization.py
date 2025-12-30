import math
import random
# Metrix ve Network modülleri, graf yapısını ve maliyet hesaplamalarını içeriyor.
from Metrix import path_metrics, total_cost
from Network import load_graph_and_demands


def aco_find_path(G, S, D, w=(1/3, 1/3, 1/3), demand = 0.0, ants = 40, iters = 40, alpha = 1.0, 
                  beta = 2.0, rho = 0.5, seed = 0):
    """
    ACO (Ant Colony Optimization) algoritması ile Kaynak (S) ve Hedef (D) arasında
    en düşük maliyetli yolu bulan fonksiyon.
    
    Yöntem: Ant System (AS) + Elitist Strateji
    Amaç: Gecikme, Güvenilirlik ve Bant Genişliği metriklerini optimize etmek.
    """

    # Tekrarlanabilir sonuçlar (reproducibility) için seed ayarı
    rnd = random.Random(seed)

    # -----------------------------------------------------------------------
    # 1) BAŞLANGIÇ (INITIALIZATION)
    # -----------------------------------------------------------------------
    # Feromon matrisi (tau) başlatılıyor. Başlangıçta tüm kenarlar eşit şansa sahiptir.
    # Bu, karıncaların başta rastgele keşif yapmasını sağlar.
    tau = {}
    for (u, v) in G.edges():
        tau[(u, v)] = 1.0
        tau[(v, u)] = 1.0

    # Feromon bırakma katsayısı (Sabit değer)
    Q = 100.0

    # -----------------------------------------------------------------------
    # 2) MALİYET VE HEURISTIC FONKSİYONU
    # -----------------------------------------------------------------------
    # Karıncaların 'gözü' (görünürlük/heuristic). 
    # Amaç: Çok kriterli problemi (Multi-objective), tek bir skalaya indirmek.
    def edge_cost(u, v):
        d = G.edges[u, v]['delay']
        lr = G.edges[u, v]['reliability'] # 0 ile 1 arasında bir olasılık değeri
        bw = G.edges[u, v]['bandwidth']

        # Reliability (Güvenilirlik) olasılıksal bir değerdir (çarpılır).
        # Ancak en kısa yol algoritmaları toplam maliyet (toplama) üzerinden çalışır.
        # -log(P) dönüşümü ile çarpma işlemini toplama işlemine çeviriyoruz.
        rel = -math.log(lr)

        # Bant genişliği (Bandwidth) ne kadar yüksekse maliyet o kadar az olmalı.
        # Bu yüzden 1/bw mantığıyla ters orantı kuruyoruz.
        res = 1000.0 / bw

        # Ağırlıklı toplam maliyet hesaplanır
        return total_cost(d, rel, res, w)

    best_path = None        # Bulunan en iyi yol (Global Best)
    best_c = float('inf')   # En iyi yolun maliyeti (Minimizasyon olduğu için sonsuz başlatılır)

    # -----------------------------------------------------------------------
    # 3) ANA DÖNGÜ (ITERATIONS)
    # -----------------------------------------------------------------------
    # Algoritma belirlenen iterasyon sayısı kadar çalışır.
    for _ in range(iters):

        paths = []  # Bu iterasyondaki tüm karıncaların yolları
        costs = []  # Bu iterasyondaki maliyetler

        iter_best_path = None        # Sadece bu iterasyonun en iyisi (Elitizm için)
        iter_best_cost = float('inf')

        # -------------------------------------------------------------------
        # 4) KARINCALARIN ÇÖZÜM İNŞASI (CONSTRUCTION)
        # -------------------------------------------------------------------
        # Her iterasyonda 'ants' sayısı kadar karınca yola çıkar.
        for _a in range(ants):
            current = S # Karınca kaynak düğümden başlar

            # Tabu Listesi: Karıncanın geçtiği düğümleri tutar, döngüleri (cycle) engeller.
            visited = {S}

            path = [S]
            steps = 0

            # Karınca hedefe ulaşana veya çıkmaz sokağa girene kadar ilerler
            while current != D and steps < len(G):

                neighbors = list(G.neighbors(current))
                rnd.shuffle(neighbors)  # Komşuları karıştır (rastgelelik faktörü)

                desirabilities = [] # Seçilme olasılıkları (Pay kısmı)
                candidates = []     # Gidilebilecek aday düğümler

                # ----------------------------------------------------------------
                # Aday Seçimi ve Kısıt Kontrolü (Constraint Handling)
                # ----------------------------------------------------------------
                for n in neighbors:

                    # 1. Kural: Zaten ziyaret edildiyse gitme (Tabu check)
                    if n in visited and n != D:
                        continue

                    # 2. Kural (QoS Kısıtı): Kenarın bant genişliği talebi karşılamıyorsa
                    # o kenarı görmezden gel (Pruning / Budama).
                    if demand > 0.0:
                        if G.edges[current, n]['bandwidth'] < demand:
                            continue

                    # ACO Formülü Bileşenleri:
                    # Tau (τ): Feromon miktarı (Geçmiş tecrübe)
                    # Eta (η): Heuristic bilgi (Anlık maliyetin tersi, 1/cost)
                    t = tau[(current, n)]
                    c = edge_cost(current, n)

                    # Görünürlük (Heuristic): Maliyet ne kadar düşükse, görünürlük o kadar yüksek.
                    eta = 1.0 / (c + 1e-6) # 0'a bölme hatasını önlemek için epsilon eklenir.

                    # Olasılık Payı Hesabı: (Feromon^alpha) * (Heuristic^beta)
                    desirabilities.append((t ** alpha) * (eta ** beta))

                    candidates.append(n)

                # Eğer gidecek uygun bir komşu yoksa (Dead-end), karınca bu turu iptal eder.
                if not candidates:
                    break

                # ----------------------------------------------------------------
                # Roulette Wheel Selection (Rulet Tekerleği Seçimi)
                # ----------------------------------------------------------------
                # Olasılıklara göre bir sonraki düğümü seçme işlemi.
                # Değeri yüksek olanın seçilme şansı artar ama garanti değildir (Stokastik).
                s = sum(desirabilities)
                r = rnd.random() * s
                acc = 0.0
                nxt = candidates[-1]

                for dval, n in zip(desirabilities, candidates):
                    acc += dval
                    if acc >= r:
                        nxt = n
                        break

                # Seçilen düğüme ilerle
                path.append(nxt)
                visited.add(nxt)
                current = nxt
                steps += 1

            # -------------------------------------------------------------------
            # 5) ÇÖZÜM DEĞERLENDİRME
            # -------------------------------------------------------------------
            # Sadece hedefe (D) ulaşabilen karıncalar değerlendirilir.
            if path[-1] == D:

                # Yolun uçtan uca darboğaz (bottleneck) kontrolü
                if demand > 0.0:
                    min_bw = min(G.edges[path[i], path[i+1]]['bandwidth']
                                 for i in range(len(path)-1))
                    if min_bw < demand:
                        continue # Talep karşılanmıyorsa çözüm geçersizdir.

                # Yolun toplam maliyetini hesapla
                d, rc, res = path_metrics(G, path)
                c = total_cost(d, rc, res, w)

                paths.append(path)
                costs.append(c)

                # Global en iyi çözümü güncelle
                if c < best_c:
                    best_c = c
                    best_path = path

                # İterasyonun en iyi çözümünü güncelle (Elitizm için gerekli)
                if c < iter_best_cost:
                    iter_best_cost = c
                    iter_best_path = path

        # -------------------------------------------------------------------
        # 6) FEROMON BUHARLAŞMASI (EVAPORATION)
        # -------------------------------------------------------------------
        # Tüm kenarlardaki feromon belirli oranda (rho) azaltılır.
        # Bu, algoritmanın eski yolları unutup yeni yollar keşfetmesini sağlar.
        for e in tau:
            tau[e] = (1 - rho) * tau[e]

        # -------------------------------------------------------------------
        # 7) FEROMON GÜNCELLEMESİ (UPDATE)
        # -------------------------------------------------------------------
        # Başarılı karıncalar geçtikleri yollara feromon bırakır.
        # Bırakılan miktar yolun kalitesiyle (1/maliyet) doğru orantılıdır.
        for path, c in zip(paths, costs):
            dep = Q / (c + 1e-6)
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                tau[(u, v)] += dep
                tau[(v, u)] += dep

        # -------------------------------------------------------------------
        # 8) ELİTİST GÜNCELLEME (ELITIST STRATEGY)
        # -------------------------------------------------------------------
        # İterasyonun en iyi karıncası (winner), geçtiği yola EKSTRA feromon bırakır.
        # Bu, algoritmanın iyi çözüme daha hızlı yakınsamasını sağlar.
        if iter_best_path is not None:
            elite_dep = Q / (iter_best_cost + 1e-6)
            for i in range(len(iter_best_path)-1):
                u, v = iter_best_path[i], iter_best_path[i+1]
                tau[(u, v)] += elite_dep
                tau[(v, u)] += elite_dep

    return best_path, path_metrics(G, best_path)

if __name__ == "__main__":
    # Veri setlerinin yüklenmesi
    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    demand_csv = "BSM307_317_Guz2025_TermProject_DemandData.csv"

    G, demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)

    # 0 numaralı düğümden 249 numaralı düğüme rota hesaplama
    best_path, metrics = aco_find_path(G, 0, 249)

    total_val = total_cost(metrics[0], metrics[1], metrics[2], (1/3, 1/3, 1/3))

    print("Best path:", best_path)
    print("Metrics (delay, reliability cost, resource cost):", metrics)
    print("Total Cost:", total_val)



    """
===========================================================================
PROJE SAVUNMASI İÇİN TEKNİK NOTLAR VE SORU-CEVAP (S.S.S)
===========================================================================

Soru 1: Neden Dijkstra gibi kesin algoritmalar yerine ACO (Sezgisel) kullandın?
---------------------------------------------------------------------------
Cevap: Problemimiz 'Çok Kriterli' (Gecikme, Güvenilirlik, Bant Genişliği) ve
yapı olarak karmaşıktır. Dijkstra tek kriterde iyidir ama bu tip NP-Zor
problemlerde ve dinamik ağ yapılarında ACO, makul sürede 'en iyiye yakın'
sonuç verdiği için tercih edilmiştir. Ayrıca yük dengeleme potansiyeli daha yüksektir.

Soru 2: Reliability (Güvenilirlik) hesabında neden -math.log() kullandın?
---------------------------------------------------------------------------
Cevap: Güvenilirlik olasılıksal bir değerdir ve seri bağlı sistemlerde 
çarpılarak hesaplanır (R_total = r1 * r2...). Ancak ACO maliyetleri 
toplayarak (C_total = c1 + c2...) ilerler. Çarpma işlemini toplama işlemine 
dönüştürmek için Logaritma kullandım. -log(R) dönüşümü ile güvenilirlik 
düştükçe ceza puanının (maliyetin) artmasını sağladım.

Soru 3: Alpha ve Beta parametreleri ne işe yarıyor?
---------------------------------------------------------------------------
Cevap: Karıncanın karar mekanizmasını dengelerler.
- Alpha (α): Feromonun (geçmiş tecrübenin) ağırlığı.
- Beta (β): Heuristic'in (anlık maliyet bilgisinin) ağırlığı.
Bu projede Beta'yı biraz daha baskın tutarak (Beta=2.0) kaliteli yollara 
öncelik verdim.

Soru 4: Elitist (Seçkinci) yaklaşım ne kattı?
---------------------------------------------------------------------------
Cevap: Her iterasyonda en iyi yolu bulan karıncaya ekstra feromon bıraktırarak
algoritmanın doğru yola daha hızlı yakınsamasını (convergence) sağladım.
Büyük graflarda (250 düğüm) çözüm süresini kısalttı.

Soru 5: Evaporation (Buharlaşma - rho) olmasa ne olurdu?
---------------------------------------------------------------------------
Cevap: Feromonlar sürekli birikir ve sonsuza giderdi. Algoritma eski ve 
muhtemelen kötü olan yolları unutamaz, yeni yollar keşfedemezdi (Local Optima).
Buharlaşma, sistemin kendini yenilemesini sağlar.
===========================================================================
"""