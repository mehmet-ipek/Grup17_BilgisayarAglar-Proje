# Sistem modüllerini içe aktarma
import sys  # Sistem parametreleri ve fonksiyonları için
import time  # Zaman ölçümleri için

# Proje içindeki özel modülleri içe aktarma
from Metrix import total_cost  # Toplam maliyet hesaplama fonksiyonu
from Network import load_graph_and_demands  # Graf ve talep verilerini yükleme fonksiyonu

# Algoritma modüllerini içe aktarma
from GeneticAlgorithm import ga_find_path  # Genetik algoritma ile yol bulma
from QLearning import q_learning_path  # Q-Learning ile yol bulma
from AntColonyOrganization import aco_find_path  # Karınca kolonisi optimizasyonu ile yol bulma

# Graf ve görselleştirme kütüphaneleri
import networkx as nx  # Graf işlemleri için NetworkX kütüphanesi
import matplotlib.pyplot as plt  # Matplotlib ile grafik çizimi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib'i PyQt5'e entegre etmek için canvas

# PyQt5 arayüz bileşenlerini içe aktarma
from PyQt5.QtWidgets import (
    QApplication,  # Ana uygulama nesnesi
    QMainWindow,  # Ana pencere sınıfı
    QWidget,  # Temel widget sınıfı
    QHBoxLayout,  # Yatay düzen yöneticisi
    QGridLayout,  # Izgara düzen yöneticisi
    QLabel,  # Metin etiketi
    QPushButton,  # Tıklanabilir buton
    QComboBox,  # Açılır liste kutusu
    QDoubleSpinBox,  # Ondalıklı sayı giriş kutusu
    QSpinBox,  # Tam sayı giriş kutusu
    QMessageBox  # İletişim kutusu
)

# Ana uygulama sınıfı - QMainWindow'dan türetilmiş
class RoutingApp(QMainWindow):
    def __init__(self, node_csv="BSM307_317_Guz2025_TermProject_NodeData.csv",
                 edge_csv="BSM307_317_Guz2025_TermProject_EdgeData.csv",
                 demand_csv="BSM307_317_Guz2025_TermProject_DemandData.csv"):
        """
        Sınıf yapıcı metodu - Uygulama başlatıldığında çalışır
        node_csv: Düğüm verilerini içeren CSV dosyası
        edge_csv: Kenar verilerini içeren CSV dosyası
        demand_csv: Talep verilerini içeren CSV dosyası
        """
        super().__init__()  # Üst sınıfın (QMainWindow) yapıcısını çağır
        
        # Pencere başlığını ayarla
        self.setWindowTitle("QoS Çok Amaçlı Rotalama - Demand Entegrasyonu (CSV)")
        # Pencere boyutunu ayarla (genişlik=1200, yükseklik=750)
        self.resize(1200, 750)

        # Graf ve talepleri CSV dosyalarından yükle
        self.G, self.demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)
        # Graf düğümlerini sıralı liste olarak al
        self.nodes = sorted(list(self.G.nodes()))
        # Varsayılan kaynak düğümü: ilk düğüm (varsa)
        self.S_default = self.nodes[0] if self.nodes else 0
        # Varsayılan hedef düğümü: son düğüm (varsa)
        self.D_default = self.nodes[-1] if self.nodes else 0
        # Senaryodan gelen bant genişliği talebi (başlangıçta 0 Mbps)
        self.current_demand = 0.0

        # Grafın görselleştirme düzenini oluştur (spring_layout algoritması)
        # seed=0: Her çalıştırmada aynı düzeni sağlar
        self.pos = nx.spring_layout(self.G, seed=0)

        # Kullanıcı arayüzü oluşturma başlangıcı
        central = QWidget()  # Merkezi widget oluştur
        self.setCentralWidget(central)  # Merkezi widget olarak ayarla
        main_layout = QHBoxLayout(central)  # Yatay ana düzen oluştur

        # Sol panel için ızgara düzeni oluştur (kontroller için)
        controls = QGridLayout()
        row = 0  # Satır sayacını başlat

        # Kaynak düğüm seçimi bölümü
        controls.addWidget(QLabel("Kaynak (S)"), row, 0)  # Etiket ekle (satır, sütun)
        self.combo_S = QComboBox()  # Açılır kutu oluştur
        # Tüm düğümleri açılır kutuya ekle
        for n in self.nodes:
            self.combo_S.addItem(str(n))
        # Varsayılan değeri ayarla
        self.combo_S.setCurrentText(str(self.S_default))
        controls.addWidget(self.combo_S, row, 1)  # Açılır kutuyu ekle
        row += 1  # Sonraki satıra geç

        # Hedef düğüm seçimi bölümü
        controls.addWidget(QLabel("Hedef (D)"), row, 0)  # Etiket ekle
        self.combo_D = QComboBox()  # Açılır kutu oluştur
        # Tüm düğümleri açılır kutuya ekle
        for n in self.nodes:
            self.combo_D.addItem(str(n))
        # Varsayılan değeri ayarla
        self.combo_D.setCurrentText(str(self.D_default))
        controls.addWidget(self.combo_D, row, 1)  # Açılır kutuyu ekle
        row += 1  # Sonraki satıra geç

        # Senaryo (Demand) seçimi bölümü
        controls.addWidget(QLabel("Senaryo (Demand)"), row, 0)  # Etiket ekle
        self.combo_dem = QComboBox()  # Açılır kutu oluştur
        # Tüm talepleri açılır kutuya ekle (kaynak->hedef (bant genişliği Mbps) formatında)
        for (s, d, b) in self.demands:
            self.combo_dem.addItem(f"{s}->{d} ({b:.0f} Mbps)")
        controls.addWidget(self.combo_dem, row, 1)  # Açılır kutuyu ekle
        row += 1  # Sonraki satıra geç

        # Senaryodan kaynak/hedef alma butonu
        self.btn_use_dem = QPushButton("Senaryodan S/D al")  # Buton oluştur
        # Butona tıklandığında on_use_demand metodunu çağır
        self.btn_use_dem.clicked.connect(self.on_use_demand)
        controls.addWidget(self.btn_use_dem, row, 0, 1, 2)  # Butonu ekle (2 sütun genişliğinde)
        row += 1  # Sonraki satıra geç

        # Algoritma seçimi bölümü
        controls.addWidget(QLabel("Algoritma"), row, 0)  # Etiket ekle
        self.combo_algo = QComboBox()  # Açılır kutu oluştur
        # Kullanılabilir algoritmaları ekle
        self.combo_algo.addItems(["GA", "Q-Learning", "ACO"])
        controls.addWidget(self.combo_algo, row, 1)  # Açılır kutuyu ekle
        row += 1  # Sonraki satıra geç

        # Ağırlıklar bölümü - Wdelay (gecikme ağırlığı)
        controls.addWidget(QLabel("Wdelay"), row, 0)  # Etiket ekle
        self.spin_wd = QDoubleSpinBox()  # Ondalıklı sayı giriş kutusu oluştur
        self.spin_wd.setRange(0.0, 1.0)  # Değer aralığını ayarla (0.0-1.0)
        self.spin_wd.setValue(1/3)  # Varsayılan değer: 1/3
        # Değer değiştiğinde check_total metodunu çağır
        self.spin_wd.valueChanged.connect(self.check_total)
        controls.addWidget(self.spin_wd, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Wreliability (güvenilirlik ağırlığı)
        controls.addWidget(QLabel("Wreliability"), row, 0)  # Etiket ekle
        self.spin_wr = QDoubleSpinBox()  # Ondalıklı sayı giriş kutusu oluştur
        self.spin_wr.setRange(0.0, 1.0)  # Değer aralığını ayarla (0.0-1.0)
        self.spin_wr.setValue(1/3)  # Varsayılan değer: 1/3
        # Değer değiştiğinde check_total metodunu çağır
        self.spin_wr.valueChanged.connect(self.check_total)
        controls.addWidget(self.spin_wr, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Wresource (kaynak ağırlığı)
        controls.addWidget(QLabel("Wresource"), row, 0)  # Etiket ekle
        self.spin_wres = QDoubleSpinBox()  # Ondalıklı sayı giriş kutusu oluştur
        self.spin_wres.setRange(0.0, 1.0)  # Değer aralığını ayarla (0.0-1.0)
        self.spin_wres.setValue(1/3)  # Varsayılan değer: 1/3
        # Değer değiştiğinde check_total metodunu çağır
        self.spin_wres.valueChanged.connect(self.check_total)
        controls.addWidget(self.spin_wres, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Genetik Algoritma parametreleri - Nesil sayısı
        controls.addWidget(QLabel("GA generations"), row, 0)  # Etiket ekle
        self.spin_gen = QSpinBox()  # Tam sayı giriş kutusu oluştur
        self.spin_gen.setRange(10, 300)  # Değer aralığını ayarla (10-300)
        self.spin_gen.setValue(90)  # Varsayılan değer: 90
        controls.addWidget(self.spin_gen, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Genetik Algoritma - Popülasyon büyüklüğü
        controls.addWidget(QLabel("GA pop size"), row, 0)  # Etiket ekle
        self.spin_pop = QSpinBox()  # Tam sayı giriş kutusu oluştur
        self.spin_pop.setRange(10, 200)  # Değer aralığını ayarla (10-200)
        self.spin_pop.setValue(60)  # Varsayılan değer: 60
        controls.addWidget(self.spin_pop, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Karınca Kolonisi Optimizasyonu - Karınca sayısı
        controls.addWidget(QLabel("ACO ants"), row, 0)  # Etiket ekle
        self.spin_ants = QSpinBox()  # Tam sayı giriş kutusu oluştur
        self.spin_ants.setRange(10, 200)  # Değer aralığını ayarla (10-200)
        self.spin_ants.setValue(40)  # Varsayılan değer: 40
        controls.addWidget(self.spin_ants, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Karınca Kolonisi Optimizasyonu - İterasyon sayısı
        controls.addWidget(QLabel("ACO iters"), row, 0)  # Etiket ekle
        self.spin_iters = QSpinBox()  # Tam sayı giriş kutusu oluştur
        self.spin_iters.setRange(10, 200)  # Değer aralığını ayarla (10-200)
        self.spin_iters.setValue(40)  # Varsayılan değer: 40
        controls.addWidget(self.spin_iters, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Pekiştirmeli Öğrenme (Q-Learning) - Bölüm sayısı
        controls.addWidget(QLabel("RL episodes"), row, 0)  # Etiket ekle
        self.spin_eps = QSpinBox()  # Tam sayı giriş kutusu oluştur
        self.spin_eps.setRange(1, 9999999)  # Değer aralığını ayarla (1-9999999)
        self.spin_eps.setValue(800)  # Varsayılan değer: 800
        controls.addWidget(self.spin_eps, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç
        
        # Pekiştirmeli Öğrenme - Epsilon değeri (keşif/sömürü dengesi)
        controls.addWidget(QLabel("RL epsilon"), row, 0)  # Etiket ekle
        self.spin_epsilon = QDoubleSpinBox()  # Ondalıklı sayı giriş kutusu oluştur
        self.spin_epsilon.setRange(0, 1)  # Değer aralığını ayarla (0-1)
        self.spin_epsilon.setValue(0.2)  # Varsayılan değer: 0.2
        controls.addWidget(self.spin_epsilon, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Manuel talep girişi (Mbps cinsinden)
        controls.addWidget(QLabel("Demand (Mbps)"), row, 0)  # Etiket ekle
        self.spin_B = QDoubleSpinBox()  # Ondalıklı sayı giriş kutusu oluştur
        self.spin_B.setRange(0.0, 1000.0)  # Değer aralığını ayarla (0-1000 Mbps)
        self.spin_B.setDecimals(1)  # Ondalık basamak sayısı: 1
        self.spin_B.setValue(0.0)  # Varsayılan değer: 0.0
        # Değer değiştiğinde update_demand_label metodunu çağır
        self.spin_B.valueChanged.connect(self.update_demand_label)
        controls.addWidget(self.spin_B, row, 1)  # Giriş kutusunu ekle
        row += 1  # Sonraki satıra geç

        # Hesaplama butonu
        self.btn_calc = QPushButton("Hesapla")  # Buton oluştur
        # Butona tıklandığında on_calculate metodunu çağır
        self.btn_calc.clicked.connect(self.on_calculate)
        controls.addWidget(self.btn_calc, row, 0, 1, 2)  # Butonu ekle (2 sütun genişliğinde)
        row += 1  # Sonraki satıra geç

        # Talep durumu etiketi
        self.lbl_demand = QLabel("Demand: (senaryodan seçin)")  # Bilgi etiketi oluştur
        controls.addWidget(self.lbl_demand, row, 0, 1, 2)  # Etiketi ekle (2 sütun genişliğinde)
        row += 1  # Sonraki satıra geç

        # Sonuç gösterim etiketi
        self.lbl_result = QLabel("Sonuçlar burada gösterilecek")  # Sonuç etiketi oluştur
        self.lbl_result.setWordWrap(True)  # Metin satır kaydırmasını etkinleştir
        controls.addWidget(self.lbl_result, row, 0, 2, 2)  # Etiketi ekle (2 satır, 2 sütun)
        row += 2  # 2 satır ilerle

        # Sol paneli oluştur ve ana düzene ekle
        left = QWidget()  # Widget oluştur
        left.setLayout(controls)  # Düzeni widget'a ata
        main_layout.addWidget(left, stretch=0)  # Sol paneli ekle (genişlemez)

        # Sağ panel: Grafik çizim alanı
        self.fig, self.ax = plt.subplots()  # Matplotlib figürü ve ekseni oluştur
        self.canvas = FigureCanvas(self.fig)  # PyQt5 canvas'ı oluştur
        main_layout.addWidget(self.canvas, stretch=1)  # Sağ paneli ekle (genişler)

        # İlk grafik çizimini yap (henüz hesaplama yapılmamış halde)
        self.draw_graph()
    
    def check_total(self):
        """
        Ağırlıkların toplamının 1.0'ı geçmemesini kontrol eder
        Toplam 1.0'ı geçerse, son değiştirilen kutudan fazlalığı düşürür
        """
        # Üç ağırlığın toplamını hesapla
        total = self.spin_wd.value() + self.spin_wr.value() + self.spin_wres.value()
        if total > 1.0:  # Toplam 1.0'dan büyükse
            # Hangi kutunun değiştirildiğini tespit et
            sender = self.sender()
            # Fazlalık miktarını hesapla
            excess = total - 1.0
            # Fazlalığı son değiştirilen kutudan düş
            sender.setValue(sender.value() - excess)
    
    def weights(self):
        """
        Ağırlıkları normalleştirilmiş tuple olarak döndürür
        Ağırlıklar toplamı 0 ise varsayılan 1/3 değerlerini kullanır
        """
        # Ağırlık değerlerini al
        wd = self.spin_wd.value()
        wr = self.spin_wr.value()
        wres = self.spin_wres.value()
        s = wd + wr + wres  # Toplamı hesapla
        
        if s == 0:  # Toplam sıfırsa
            # Kullanıcıyı uyar
            QMessageBox.warning(self, "Uyarı", "Ağırlıkların toplamı 0 olamaz. Varsayılan 1/3 uygulanacak.")
            return (1/3, 1/3, 1/3)  # Varsayılan değerleri döndür
        
        # Normalleştirilmiş ağırlıkları döndür (toplam 1.0 olacak şekilde)
        return (wd/s, wr/s, wres/s)

    def draw_graph(self, path=None, title=""):
        """
        Ağ grafiğini çizer
        path: Vurgulanacak yol (opsiyonel)
        title: Grafik başlığı
        """
        self.ax.clear()  # Önceki çizimi temizle
        
        # Tüm düğümleri gri renkte çiz
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=77, node_color='lightgray')
        # Tüm kenarları yarı saydam gri renkte çiz
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='lightgray', width=0.5, alpha=0.3)
        # Düğüm etiketlerini çiz
        nx.draw_networkx_labels(self.G, self.pos, labels={n: str(n) for n in self.G.nodes()}, font_size=8, font_color="black", ax=self.ax)

        # Eğer bir yol verilmişse ve en az 2 düğüm içeriyorsa
        if path and len(path) > 1:
            # Yol kenarlarını oluştur (ardışık düğüm çiftleri)
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            # Ara düğümleri (başlangıç ve bitiş hariç) turuncu renkle çiz
            if len(path) > 2:
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=path[1:-1], ax=self.ax, node_size=100, node_color='orange')
            
            # Başlangıç düğümünü açık yeşil renkle çiz
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[0]], ax=self.ax, node_size=130, node_color='lightgreen')
            # Bitiş düğümünü mavi renkle çiz
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[path[-1]], ax=self.ax, node_size=130, node_color='deepskyblue')
            # Yol kenarlarını kırmızı renkle kalın çiz
            nx.draw_networkx_edges(self.G, self.pos, edgelist=path_edges, ax=self.ax, edge_color='red', width=2.5)

        # Grafik başlığını ayarla
        self.ax.set_title(title)
        # Eksenleri gizle
        self.ax.set_axis_off()
        # Düzeni optimize et
        self.fig.tight_layout()
        # Çizimi güncelle
        self.canvas.draw()

    def on_use_demand(self):
        """
        Senaryodan kaynak, hedef ve talep değerlerini alır
        Seçili senaryoya göre S, D ve talep değerlerini günceller
        """
        # Seçili senaryo indeksini al
        idx = self.combo_dem.currentIndex()
        # İndeks geçersizse uyar
        if idx < 0 or idx >= len(self.demands):
            QMessageBox.warning(self, "Uyarı", "Senaryo seçimi geçersiz.")
            return
        
        # Senaryo verilerini al (kaynak, hedef, bant genişliği)
        s, d, b = self.demands[idx]
        # Kaynak düğümü combo box'ını güncelle
        self.combo_S.setCurrentText(str(s))
        # Hedef düğümü combo box'ını güncelle
        self.combo_D.setCurrentText(str(d))
        # Mevcut talep değerini kaydet
        self.current_demand = b
        # Talep giriş kutusunu güncelle
        self.spin_B.setValue(b)
        # Talep etiketini güncelle
        self.lbl_demand.setText(f"Demand: {b:.1f} Mbps (senaryodan)")

        # Kullanıcıyı bilgilendir
        QMessageBox.information(self, "Bilgi", f"Senaryodan alındı: S={s}, D={d}, talep={b:.0f} Mbps")
        
    def update_demand_label(self, text):
        """
        Talep giriş kutusundaki değer değiştiğinde etiket güncellenir
        text: Yeni talep değeri (string olarak gelir)
        """
        val = float(text)  # String'i float'a çevir
        
        # Eğer talep 0 ise
        if val == 0.0:
            self.lbl_demand.setText("Demand: (senaryodan seçin)")
        else:  # Talep 0'dan büyükse
            self.lbl_demand.setText(f"Demand: {val:.1f} Mbps")

    def on_calculate(self):
        """
        Hesapla butonuna basıldığında çalışır
        Seçilen algoritma ile en iyi yolu bulur ve sonuçları görselleştirir
        """
        # Kaynak ve hedef düğümlerini al
        try:
            S = int(self.combo_S.currentText())  # Kaynak düğümü
            D = int(self.combo_D.currentText())  # Hedef düğümü
        except Exception:
            # Hata durumunda kullanıcıyı bilgilendir
            QMessageBox.critical(self, "Hata", "S/D seçiminde sorun var.")
            return

        # Kaynak ve hedef aynı olamaz kontrolü
        if S == D:
            QMessageBox.warning(self, "Uyarı", "Kaynak ve hedef farklı olmalı.")
            return

        # Normalleştirilmiş ağırlıkları al
        w = self.weights()

        # Seçili algoritmayı al
        algo = self.combo_algo.currentText()

        # Kullanılacak talep değerini belirle:
        # Manuel giriş > 0 ise onu kullan, değilse senaryodan gelen değeri kullan
        demand_used = self.spin_B.value() if self.spin_B.value() > 0.0 else self.current_demand
    
        # Yol bulma işleminin başlangıç zamanını kaydet
        t0 = time.time()
        
        # Başlangıç değerleri
        path, metrics = None, (0.0, 0.0, 0.0)
        
        # Seçilen algoritmaya göre yol bul
        if algo == "GA":  # Genetik Algoritma
            path, metrics = ga_find_path(
                self.G, S, D, w=w,
                demand=demand_used,  # Talep parametresi
                generations=self.spin_gen.value(),  # Nesil sayısı
                pop_size=self.spin_pop.value(),  # Popülasyon boyutu
                seed=0  # Rastgelelik tohumu
            )
        elif algo == "ACO":  # Karınca Kolonisi Optimizasyonu
            path, metrics = aco_find_path(
                self.G, S, D, w=w,
                demand=demand_used,  # Talep parametresi
                ants=self.spin_ants.value(),  # Karınca sayısı
                iters=self.spin_iters.value(),  # İterasyon sayısı
                seed=0  # Rastgelelik tohumu
            )
        else:  # Q-Learning
            path, metrics, Q = q_learning_path(
                self.G, S, D,
                w=w,
                demand=demand_used,  # Talep parametresi
                episodes=self.spin_eps.value(),  # Bölüm sayısı
                alpha=0.2,  # Öğrenme oranı
                gamma=0.95,  # İndirim faktörü
                epsilon=self.spin_epsilon.value(),  # Keşif-sömürü dengesi
                seed=42  # Rastgelelik tohumu
            )
    
        # Algoritma çalışma süresini hesapla
        rt_alg = time.time() - t0
        # Metrikleri ayır (gecikme, güvenilirlik maliyeti, kaynak maliyeti)
        d, rc, res = metrics
        # Toplam maliyeti hesapla
        cost = total_cost(d, rc, res, w)
        
        # Kapasite kontrolü için değişkenler
        capacity_txt = ""  # Kapasite durumu mesajı
        bandwidth_warning = False  # Uyarı bayrağı

        # Yol boyunca minimum bant genişliğini bul
        bw_values = []  # Bant genişliği değerleri listesi
        # Yoldaki her kenar için
        for i in range(len(path)-1):
            # Kenar varsa bant genişliğini al
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

        # Kapasite yeterli mi kontrol et
        if demand_used > 0.0:
            if min_bw < demand_used:
                QMessageBox.warning(
                    self, "Kapasite Uyarısı",
                    f"UYARI: {S} → {D} arasında talebi karşılayan kenar bulunamadı!"
                )
                return
            else:
                capacity_txt = f"\n✓ Kapasite uygun: min bw={min_bw:.1f} Mbps, talep={demand_used:.1f} Mbps"

        # Grafik başlığını oluştur
        title = f"{algo} yolu"
        if bandwidth_warning:
            title += " ⚠️ (Talebi karşılamıyor)"

        # Grafik çiz ve sonuçları göster
        self.draw_graph(path, title=title)

        result_text = (f"Algoritma={algo}, S={S}, D={D}\n"
                       f"Delay={d:.3f} ms, RelCost={rc:.4f}, ResCost={res:.4f}\n"
                       f"TotalCost={cost:.4f}, Runtime={rt_alg:.2f} s")

        if demand_used > 0.0:
            result_text += f"\nTalep: {demand_used:.1f} Mbps"

        result_text += f"{capacity_txt}"

        self.lbl_result.setText(result_text)


if __name__ == "__main__":
    # Uygulamayı başlat
    app = QApplication(sys.argv)
    win = RoutingApp(
        node_csv="BSM307_317_Guz2025_TermProject_NodeData.csv",
        edge_csv="BSM307_317_Guz2025_TermProject_EdgeData.csv",
        demand_csv="BSM307_317_Guz2025_TermProject_DemandData.csv"
    )
    win.show()
    sys.exit(app.exec_())