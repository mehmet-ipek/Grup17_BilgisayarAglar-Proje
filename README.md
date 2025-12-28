# QoS Odaklı Çok Amaçlı Rotalama için Meta-Sezgisel ve Pekiştirmeli Öğrenme Yaklaşımları

## 📌 Proje Özeti
**Ders:** BSM307 - Algoritma Analizi ve Tasarımı (Güz 2025)

Bu proje, **250 düğümlü (node)** ve karmaşık bağlantılara sahip rastgele oluşturulmuş bir ağ topolojisi üzerinde, kaynak (S) ile hedef (D) arasında **en optimum** yolu bulmayı amaçlayan bir optimizasyon çalışmasıdır.

"En uygun yol" kavramı tek bir kritere göre değil, birbiriyle çelişebilen üç farklı QoS (Quality of Service) metriği üzerinden tanımlanmıştır:
1.  ⏱ **Toplam Gecikme (Delay):** Minimize edilir.
2.  🔒 **Güvenilirlik (Reliability):** Maksimize edilir.
3.  📡 **Ağ Kaynak Kullanımı (Bandwidth):** Minimize edilir.

Problem **NP-Hard** sınıfında olduğu için deterministik algoritmalar yerine **Meta-Sezgisel (GA, ACO)** ve **Pekiştirmeli Öğrenme (Q-Learning)** tabanlı yaklaşımlar geliştirilmiş ve kıyaslanmıştır.

---

## 🌐 Ağ Modeli ve Topoloji

Proje, gerçekçi bir ağ altyapısını simüle etmek için aşağıdaki graf teorisi modelini kullanır:

* **Topoloji:** Erdős–Rényi $G(n, p)$ modeli.
* **Düğüm Sayısı ($N$):** 250
* **Bağlantı Olasılığı ($P$):** 0.4
* **Özellik:** Grafiğin bağlı (connected) olduğu garanti altına alınmıştır.

### Düğüm ve Kenar Özellikleri
| Bileşen | Özellik | Açıklama |
|---|---|---|
| **Düğüm** | Processing Delay | Ara düğümlerdeki işlem gecikmesi. |
| **Düğüm** | Node Reliability | Düğümün arızalanmama olasılığı. |
| **Kenar** | Bandwidth | Mbps cinsinden bant genişliği kapasitesi. |
| **Kenar** | Link Delay | İletim hattındaki gecikme (ms). |
| **Kenar** | Link Reliability | Hattın kopmama olasılığı. |

---

## ⚙️ Matematiksel Model ve QoS Metrikleri

Üç çelişen kriterin **Ağırlıklı Toplamı (Weighted Sum Model)** minimize edilerek tek bir amaç fonksiyonuna indirgenmiştir:

### 1. Toplam Gecikme (Minimize)
$$\text{Delay}(P) = \sum_{(i,j) \in P} \text{LinkDelay}_{ij} + \sum_{k \in P} \text{ProcessingDelay}_k$$

### 2. Toplam Güvenilirlik (Maksimize -> Minimize Dönüşümü)
Güvenilirlik çarpımsal bir metriktir. İşlem kolaylığı ve sayısal kararlılık (underflow önleme) için **Logaritmik Dönüşüm** uygulanarak toplamsal maliyete çevrilmiştir.
$$\text{ReliabilityCost}(P) = \sum_{(i,j) \in P} [-\log(R_{link})] + \sum_{k \in P} [-\log(R_{node})]$$

### 3. Ağ Kaynak Kullanımı (Minimize)
Yüksek bant genişliğine sahip yollar teşvik edilir.
$$\text{ResourceCost}(P) = \sum_{(i,j) \in P} \left( \frac{1000}{\text{Bandwidth}_{ij}} \right)$$

### 🎯 Amaç Fonksiyonu (Fitness)
$$\text{TotalCost} = (W_d \times \text{Delay}) + (W_r \times \text{RelCost}) + (W_{res} \times \text{ResCost})$$

---

## 🚀 Kullanılan Algoritmalar ve Teknik Detaylar

### 🧬 1. Genetik Algoritma (GA)
Klasik Shortest Path algoritmaları kullanılmadan, tamamen evrimsel süreçle çalışan özelleştirilmiş bir yapıdır.

* **Birey:** Kaynaktan hedefe giden bir yol (Path Representation).
* **Başlangıç Popülasyonu:** `Random Walk` (Rastgele Yürüyüş) tabanlı üretim.
* **Seçilim (Selection):** Rank-based roulette wheel (Seçilim baskısı düşürülerek çeşitlilik korunur).
* **Çaprazlama (Crossover):** *Path-aware crossover* (İki yolun ortak düğümlerinden birleşmesi).
* **Mutasyon:**
    * *Rastgele Gen Değişimi:* Tek bir düğümün değişmesi.
    * *Segment Reset:* Yolun bir kısmının silinip rastgele yeniden oluşturulması.
* **Çeşitlilik Koruma (Fitness Sharing):** Klon bireyler silinmez, ancak sayıları arttıkça fitness değerleri kötüleştirilerek avantajları kırılır. **Amaç:** Erken yakınsamayı (premature convergence) önlemektir.

### 🐜 2. Karınca Kolonisi Optimizasyonu (ACO)
* **Yöntem:** Klasik Ant System (AS) + Elitist Strateji.
* **Geçiş Olasılığı:** Feromon ($\tau$) + Heuristic ($\eta$) kombinasyonu.
* **Heuristic Bilgi:** Kenar maliyetinin tersi ($1 / \text{Cost}$).
* **QoS Kısıtı:** Bant genişliği talebini ($BW \ge Demand$) karşılamayan kenarlar budanır (pruning).
* **Elitizm:** Her iterasyonda en iyi yolu bulan karınca ekstra feromon bırakır.

### 🤖 3. Q-Learning (Reinforcement Learning)
* **Durum (State):** Mevcut düğüm.
* **Aksiyon (Action):** Komşu düğüme geçiş.
* **Ödül (Reward):** Düşük maliyet $\to$ Yüksek ödül. Hedefe ulaşınca ekstra bonus.
* **Politika:** $\epsilon$-Greedy (Keşif ve Sömürü dengesi).
* **İlklendirme:** Q-Tablosu, kenar maliyetleri ile ön-ilklendirilerek öğrenme süresi kısaltılmıştır.

---

## 🖥️ Uygulama Arayüzü (GUI) Özellikleri

`App.py` dosyası ile çalışan PyQt5 arayüzü şu özellikleri sunar:

1.  **Dinamik Grafik Çizimi:** Seçilen kaynak ve hedef düğümler ile bulunan yol grafik üzerinde renklendirilerek gösterilir.
2.  **Senaryo Entegrasyonu:** `DemandData.csv` dosyasındaki hazır senaryolar listeden seçilebilir.
3.  **Parametre Ayarı:** Ağırlıklar ($W_{delay}, W_{rel}, W_{res}$), Popülasyon, İterasyon, Epsilon vb.
4.  **Kapasite Kontrolü:** Bulunan yolun talep edilen bant genişliğini karşılayıp karşılamadığı otomatik kontrol edilir.

---

## 📂 Proje Dosya Yapısı
BSM307_Project/
│
├── App.py
├── Network.py
├── Metrix.py
├── GeneticAlgorithm.py
├── AntColonyOrganization.py
├── QLearning.py
│
├── BSM307_317_Guz2025_TermProject_NodeData.csv
├── BSM307_317_Guz2025_TermProject_EdgeData.csv
├── BSM307_317_Guz2025_TermProject_DemandData.csv
│
├── requirements.txt
└── README.md

## 👥 Görev Dağılımı

| Ekip Üyesi | Sorumluluk Alanı |
|-----------|------------------|
| **Efe Baykın** | Ağ modeli, topoloji tasarımı, düğüm ve bağlantı özelliklerinin tanımlanması (`Network.py`) |
| **Maysam Wazin** | Optimizasyon problemi tanımı, QoS metriklerinin matematiksel modellenmesi ve ağırlıklı maliyet fonksiyonu (`Metrix.py`) |
| **Betül Korkmaz** | Genetik Algoritma (GA) geliştirilmesi |
| **Mehmet İpek** | Genetik Algoritma (GA) geliştirilmesi |
| **Murat Süleymanoğlu** | Karınca Kolonisi Optimizasyonu (ACO) algoritmasının geliştirilmesi (`AntColonyOrganization.py`) |
| **Furkan Çat** | Pekiştirmeli Öğrenme tabanlı Q-Learning algoritmasının tasarımı ve uygulanması (`QLearning.py`) |
| **Ahmet Nihat Karkaç** | Grafiksel kullanıcı arayüzü (GUI) geliştirilmesi (`App.py`) |
| **Aytekin Yılmaz** | Grafiksel kullanıcı arayüzü (GUI) geliştirilmesi (`App.py`) |


---

## 🛠️ Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python App.py

## 🌱 Seed ve Tekrarlanabilirlik

- **Varsayılan Seed:** `0`

Aynı seed kullanıldığında:
- Aynı ağ çizimi elde edilir
- Aynı başlangıç popülasyonu oluşturulur
- Deney sonuçları tekrarlanabilir (reproducible) olur

Seed değeri değiştirilerek farklı rastgelelik senaryoları test edilebilir.

## 📊 Deneysel Değerlendirme

Tüm algoritmalar:
- Aynı ağ topolojisi
- Aynı QoS ağırlıkları
- Aynı demand senaryoları  

altında test edilmiştir.

**Ölçülen kriterler:**
- Toplam maliyet (Total Cost)
- Çalışma süresi (Runtime)
- Yol uzunluğu (Hop sayısı)
