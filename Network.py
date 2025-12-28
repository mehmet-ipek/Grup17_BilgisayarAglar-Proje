import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def _to_float(x):
    """
    CSV verilerindeki ondalık ayırıcı sorununu çözer.
    Bazı sistemlerde (TR) virgül (,) kullanılırken Python nokta (.) bekler.
    Bu fonksiyon ',' karakterini '.' ile değiştirip float'a çevirir.
    """
    if isinstance(x, str):
        x = x.replace(",", ".")
    return float(x)

def load_graph_and_demands(node_path="BSM307_317_Guz2025_TermProject_NodeData.csv",
                           edge_path="BSM307_317_Guz2025_TermProject_EdgeData.csv",
                           demand_path="BSM307_317_Guz2025_TermProject_DemandData.csv"):
    """
    Verilen CSV dosyalarını okur ve NetworkX graf yapısını oluşturur.
    
    Çıktı:
        G: Düğümlerin ve kenarların yüklendiği NetworkX graf objesi.
        demands: (Kaynak, Hedef, İstenen_Bant_Genişliği) formatında talep listesi.
    """
    
    # -----------------------------------------------------------
    # 1. ADIM: Düğüm (Node) Verilerinin Okunması ve Temizlenmesi
    # -----------------------------------------------------------
    # Pandas ile CSV okunur, ayırıcı (separator) olarak ';' kullanılır.
    node_df = pd.read_csv(node_path, sep=";")
    
    # Sayısal değerler string gelebileceği için float dönüşümü yapılır.
    node_df["s_ms"] = node_df["s_ms"].apply(_to_float)    # İşlem gecikmesi (processing delay)
    node_df["r_node"] = node_df["r_node"].apply(_to_float) # Güvenilirlik (reliability)

    # -----------------------------------------------------------
    # 2. ADIM: Kenar (Edge/Link) Verilerinin Okunması
    # -----------------------------------------------------------
    edge_df = pd.read_csv(edge_path, sep=";")
    edge_df["capacity_mbps"] = edge_df["capacity_mbps"].apply(_to_float) # Bant genişliği
    edge_df["delay_ms"] = edge_df["delay_ms"].apply(_to_float)           # İletim gecikmesi
    edge_df["r_link"] = edge_df["r_link"].apply(_to_float)               # Hat güvenilirliği

    # -----------------------------------------------------------
    # 3. ADIM: Talep (Demand) Verilerinin Okunması
    # -----------------------------------------------------------
    demand_df = pd.read_csv(demand_path, sep=";")
    demand_df["demand_mbps"] = demand_df["demand_mbps"].apply(_to_float)

    # -----------------------------------------------------------
    # 4. ADIM: NetworkX Grafının İnşası (Topology Building)
    # -----------------------------------------------------------
    G = nx.Graph() # Yönsüz (Undirected) graf oluşturuluyor.

    # Düğümleri ve özelliklerini ekle
    for _, r in node_df.iterrows():
        nid = int(r["node_id"])
        # Node'a ait özellikleri (attribute) sözlük yapısında saklıyoruz.
        G.add_node(nid, proc_delay=float(r["s_ms"]), reliability=float(r["r_node"]))

    # Kenarları ve özelliklerini ekle
    for _, r in edge_df.iterrows():
        u, v = int(r["src"]), int(r["dst"])
        # Kenar ağırlıkları ve QoS parametreleri burada tanımlanır.
        G.add_edge(u, v,
                   bandwidth=float(r["capacity_mbps"]),
                   delay=float(r["delay_ms"]),
                   reliability=float(r["r_link"]))

    # Talepleri tuple listesi haline getir: (Kaynak, Hedef, Miktar)
    demands = [(int(r["src"]), int(r["dst"]), float(r["demand_mbps"])) for _, r in demand_df.iterrows()]
    
    return G, demands

# =========================
# Örnek kullanım (Test Bloğu)
# =========================
if __name__ == "__main__":
    # Dosya yolları tanımlanır
    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    demand_csv = "BSM307_317_Guz2025_TermProject_DemandData.csv"

    # Fonksiyon çağrılır ve graf yüklenir
    try:
        G, demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)

        print(f"Graf Başarıyla Oluşturuldu.")
        print(f"Toplam Düğüm Sayısı: {G.number_of_nodes()}")
        print(f"Toplam Kenar Sayısı: {G.number_of_edges()}")
        print(f"Yüklenen Talep Sayısı: {len(demands)}")
        
        # -----------------------------------------------------------
        # GÖRSELLEŞTİRME (VISUALIZATION)
        # -----------------------------------------------------------
        # Spring Layout: Düğümleri yaylı fizik modeline göre yerleştirir.
        # seed=0 parametresi her çalıştırmada şeklin aynı kalmasını sağlar.
        pos = nx.spring_layout(G, seed=0) 
        
        plt.figure(figsize=(10, 8)) # Çizim alanı boyutu
        nx.draw(G, pos,
                with_labels=True,        # Düğüm numaralarını göster
                node_color="#A0CBE2",    # Düğüm rengi (açık mavi)
                node_size=300,           # Düğüm boyutu
                font_size=8,             # Yazı boyutu
                edge_color="gray",       # Kenar rengi
                width=0.5)               # Kenar kalınlığı
        
        plt.title("Network Topolojisi Görselleştirmesi")
        plt.show() # Grafiği ekrana bas

    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı! Lütfen dosya yollarını kontrol edin.\n{e}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu:\n{e}")


"""
===========================================================================
PROJE SAVUNMASI İÇİN TEKNİK NOTLAR VE SORU-CEVAP (S.S.S)
===========================================================================

ÖZET SUNUM METNİ:
"Hocam, bu modül projenin 'Veri Katmanı'dır (Data Layer). Pandas kütüphanesi
ile CSV formatındaki topoloji verilerini okuyup, bunları NetworkX kütüphanesi
üzerinde matematiksel bir graf yapısına dönüştürüyorum. Ayrıca veri setindeki
bölgesel format farklılıklarını (virgül/nokta) düzelten bir ön işleme
(preprocessing) mekanizması da ekledim. Bu yapı, optimizasyon algoritmama
(ACO) girdi olarak verilmektedir."

Soru 1: Neden NetworkX kütüphanesini tercih ettin?
---------------------------------------------------------------------------
Cevap: NetworkX, Python'da graf teorisi algoritmaları için endüstri standardıdır.
Düğüm ve kenarlara (Nodes/Edges) dinamik olarak 'attribute' (gecikme, bant
genişliği gibi) eklemeyi çok kolaylaştırıyor. Kendi yazdığım graf yapısına
göre hem daha optimize çalışıyor hem de görselleştirme araçları sunuyor.

Soru 2: '_to_float' fonksiyonuna neden ihtiyaç duydun?
---------------------------------------------------------------------------
Cevap: Veri setlerimiz (CSV) bazen Türkçe yerel ayarlarıyla (ondalık ayırıcı
olarak virgül) kaydedilmiş olabiliyor. Python float dönüşümü için nokta (.)
bekler. Bu fonksiyon, veri temizleme (data cleaning) aşamasında format
uyumsuzluğundan kaynaklanacak hataları önlemek için yazıldı.

Soru 3: spring_layout nedir? Grafiği neye göre çiziyor?
---------------------------------------------------------------------------
Cevap: Düğümlerin ekranda rastgele değil, okunabilir şekilde dağılması için
kullandığım bir algoritmadır (Force-directed graph drawing). Düğümleri iten,
kenarları çeken yaylar varmış gibi simüle eder. Topolojinin genel yapısını
ve kopukluk olup olmadığını gözle kontrol etmemi (Debugging) sağladı.

Soru 4: Grafın Yönlü (Directed) mü, Yönsüz (Undirected) mü?
---------------------------------------------------------------------------
Cevap: 'nx.Graph()' kullanarak Yönsüz (Undirected) bir graf oluşturdum.
Çünkü ağ topolojilerinde fiziksel kablolar genellikle çift yönlü iletişime
izin verir (Full Duplex). A'dan B'ye hat varsa, B'den A'ya da aynı hat vardır.
===========================================================================
"""