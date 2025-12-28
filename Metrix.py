import math
# math kütüphanesi matematiksel işlemler için kullanılır
# Bu projede özellikle logaritma (log) fonksiyonuna ihtiyaç vardır
# Güvenilirlik hesaplarında -log(reliability) kullandığımız için zorunludur

from Network import load_graph_and_demands
# Network.py dosyasından load_graph_and_demands fonksiyonu içe aktarılır
# Bu fonksiyon:
# - Düğümleri (nodes)
# - Kenarları (edges)
# - Gecikme, güvenilirlik, bant genişliği gibi ağ parametrelerini
# CSV dosyalarından okuyup bir NetworkX grafı (G) oluşturur


# =========================
# Metrikler ve toplam maliyet
# =========================
# Bu dosya bilgisayar ağlarında bir yolun (path) performansını ölçmek için yazılmıştır
# Hesaplanan metrikler:
# 1) Toplam gecikme (Delay)
# 2) Güvenilirlik maliyeti (Reliability Cost)
# 3) Ağ kaynak kullanımı (Resource Cost)


def path_metrics(G, path):
    # Bu fonksiyon verilen bir yol (path) için ağ metriklerini hesaplar
    # G    : NetworkX graph (ağ yapısı)
    # path : Kaynak düğümden hedef düğüme giden düğüm listesi (örn: [0, 2, 5])

    if not path or len(path) < 2:
        # Eğer yol boşsa veya tek düğümden oluşuyorsa
        # Bu geçersiz bir yol demektir
        # Sonsuz maliyet döndürülerek bu yolun seçilmesi engellenir
        return float('inf'), float('inf'), float('inf')


    # Kenar gecikmesi
    delay_edges = sum(
        G.edges[path[i], path[i+1]]['delay']
        for i in range(len(path)-1)
    )
    # Yol üzerindeki her kenarın (linkin) gecikmesi toplanır
    # Bu gecikmeler:
    # - İletim gecikmesi
    # - Yayılma gecikmesi
    # gibi ağ gecikmelerini temsil eder


    # Ara düğüm işlem gecikmesi (S ve D hariç)
    delay_nodes = sum(
        G.nodes[n]['proc_delay']
        for n in path[1:-1]
    )
    # Kaynak (S) ve hedef (D) düğüm hariç tutulur
    # Ara düğümlerdeki işlem gecikmeleri hesaplanır
    # Bu, router/switch üzerinde paketin işlenme süresini temsil eder


    delay = delay_edges + delay_nodes
    # Toplam gecikme:
    # Kenar gecikmeleri + düğüm işlem gecikmeleri
    # Uçtan uca gecikmeyi temsil eder


    # Güvenilirlik maliyeti = -log(kenar) + -log(düğüm)
    # Normalde güvenilirlikler çarpılır
    # -log kullanılarak:
    # - Çarpım → toplama dönüşür
    # - Sayısal kararlılık sağlanır
    # Bu yöntem ağ literatüründe yaygındır


    rel_cost_edges = sum(
        -math.log(G.edges[path[i], path[i+1]]['reliability'])
        for i in range(len(path)-1)
    )
    # Yol üzerindeki her kenarın güvenilirliği alınır
    # Güvenilirlik düştükçe maliyet artar
    # Daha az güvenilir link = daha pahalı link


    rel_cost_nodes = sum(
        -math.log(G.nodes[n]['reliability'])
        for n in path
    )
    # Düğümlerin güvenilirliği de hesaba katılır
    # Çünkü gerçek ağlarda router/switch arızaları da iletişimi etkiler


    rel_cost = rel_cost_edges + rel_cost_nodes
    # Toplam güvenilirlik maliyeti
    # Kenar + düğüm güvenilirlik maliyetleri


    # Kaynak maliyeti = sum(1000 / bandwidth)
    res_cost = sum(
        1000.0 / G.edges[path[i], path[i+1]]['bandwidth']
        for i in range(len(path)-1)
    )
    # Bant genişliği ne kadar düşükse kaynak o kadar kıttır
    # Bu yüzden ters orantı kullanılır (1 / bandwidth)
    # 1000 değeri ölçeklendirme (normalizasyon) içindir


    return delay, rel_cost, res_cost
    # Fonksiyon sırasıyla:
    # - Toplam gecikme
    # - Güvenilirlik maliyeti
    # - Kaynak maliyetini döndürür


def total_cost(delay, rel_cost, res_cost, w):
    # Bu fonksiyon farklı metrikleri tek bir toplam maliyette birleştirir
    # Çok amaçlı (multi-objective) optimizasyon yaklaşımıdır

    wd, wr, wres = w
    # wd   : gecikme ağırlığı
    # wr   : güvenilirlik ağırlığı
    # wres : kaynak maliyeti ağırlığı

    return wd * delay + wr * rel_cost + wres * res_cost
    # Ağırlıklı toplam maliyet hesaplanır
    # Ağırlıklar değiştirilerek farklı QoS senaryoları oluşturulabilir


if __name__ == "__main__":
    # Bu blok dosya doğrudan çalıştırıldığında test amaçlı çalışır

    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    # Düğüm bilgilerini içeren CSV dosyası

    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    # Kenar bilgilerini (delay, reliability, bandwidth) içeren CSV dosyası

    demand_csv = "BSM307_317_Guz2025_TermProject_DemandData.csv"
    # Trafik taleplerini içeren CSV dosyası


    G, demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)
    # CSV dosyaları okunur
    # NetworkX graph (G) ve talepler oluşturulur


    # Yol tanımla
    path = [0, 2]
    # Örnek bir yol seçilmiştir
    # Sunum ve test amaçlıdır


    # Metrikleri hesapla
    delay, rel_cost, res_cost = path_metrics(G, path)
    # Seçilen yol için tüm metrikler hesaplanır


    total = total_cost(delay, rel_cost, res_cost, (1/3, 1/3, 1/3))
    # Üç metriğe eşit ağırlık verilmiştir
    # Hiçbiri diğerinden daha önemli kabul edilmemiştir


    print("Path:", path)
    # Kullanılan yol ekrana yazdırılır

    print("Delay:", delay)
    # Toplam gecikme değeri yazdırılır

    print("Reliability cost:", rel_cost)
    # Güvenilirlik maliyeti yazdırılır

    print("Resource cost:", res_cost)
    # Ağ kaynak maliyeti yazdırılır

    print("Total cost:", total)
    # Ağırlıklı toplam maliyet yazdırılır

# =========================
# HOCANIN SORABİLECEĞİ OLASI SORULAR VE CEVAPLAR
# =========================

# SORU 1:
# Bu projede neden gecikme (delay) metriğini kullandınız?
# CEVAP:
# Bilgisayar ağlarında performansın en temel göstergelerinden biri uçtan uca gecikmedir.
# Gerçek zamanlı uygulamalar (video, ses, online oyunlar) gecikmeye çok duyarlıdır.
# Bu yüzden yol seçiminde mutlaka hesaba katılması gerekir.


# SORU 2:
# Kenar gecikmesi ile düğüm gecikmesini neden ayrı ayrı hesapladınız?
# CEVAP:
# Çünkü ağlarda gecikme sadece linklerden kaynaklanmaz.
# Router ve switch'lerde paket işleme süresi de gecikmeye sebep olur.
# Gerçekçi bir ağ modeli oluşturmak için her ikisi ayrı ayrı modellenmiştir.


# SORU 3:
# Güvenilirlik neden doğrudan kullanılmadı da -log(reliability) kullanıldı?
# CEVAP:
# Güvenilirlik değerleri normalde çarpılarak hesaplanır.
# Ancak çarpım işlemi sayısal olarak kararsızdır.
# -log kullanılarak çarpım toplama çevrilir ve optimizasyon daha stabil hale gelir.
# Bu yöntem ağ literatüründe standart bir yaklaşımdır.


# SORU 4:
# Neden sadece link güvenilirliği değil düğüm güvenilirliği de dahil edildi?
# CEVAP:
# Gerçek ağlarda sadece linkler değil router ve switch'ler de arızalanabilir.
# Bu nedenle düğümlerin güvenilirliği de uçtan uca iletişimi doğrudan etkiler.
# Daha gerçekçi bir model elde etmek için düğüm güvenilirliği eklenmiştir.


# SORU 5:
# Kaynak maliyeti neden 1 / bandwidth şeklinde tanımlandı?
# CEVAP:
# Bant genişliği ne kadar düşükse kaynak o kadar kıt ve değerlidir.
# Bu nedenle bant genişliği ile ters orantılı bir maliyet tanımı yapılmıştır.
# Düşük bandwidth = yüksek maliyet mantığı kullanılmıştır.


# SORU 6:
# 1000.0 sayısı neden kullanıldı?
# CEVAP:
# Bu değer ölçeklendirme (normalizasyon) amacıyla kullanılmıştır.
# Amaç maliyet değerlerinin çok küçük çıkmasını engellemek
# ve metrikleri aynı büyüklük mertebesine yaklaştırmaktır.


# SORU 7:
# Toplam maliyet fonksiyonunda neden ağırlıklar (w) kullanıldı?
# CEVAP:
# Çünkü bu problem çok amaçlı bir optimizasyon problemidir.
# Farklı senaryolarda:
# - Gecikme daha önemli olabilir
# - Güvenilirlik daha önemli olabilir
# Ağırlıklar sayesinde bu senaryolar kolayca modellenebilir.


# SORU 8:
# Ağırlıkları neden (1/3, 1/3, 1/3) seçtiniz?
# CEVAP:
# Bu çalışmada tüm metriklerin eşit öneme sahip olduğu varsayılmıştır.
# Böylece hiçbir metrik diğerine baskın hale gelmemiştir.
# Bu seçim test ve karşılaştırma amacıyla yapılmıştır.


# SORU 9:
# Bu metrikler hangi ders konularıyla ilişkilidir?
# CEVAP:
# QoS (Quality of Service),
# Ağ performans analizi,
# Çok amaçlı optimizasyon,
# Yönlendirme algoritmaları konularıyla doğrudan ilişkilidir.


# SORU 10:
# Bu metrik yapısı gerçek hayatta nerelerde kullanılır?
# CEVAP:
# MPLS ağları,
# Yazılım tanımlı ağlar (SDN),
# Trafik mühendisliği (Traffic Engineering),
# Kritik altyapı ağları gibi sistemlerde kullanılır.



#ben bu kısmı QoS tabanlı ağ performans analizi olarak ele aldım.
#Gecikme, güvenilirlik ve kaynak kullanımını ayrı ayrı modelleyip
#ağırlıklı toplam maliyet fonksiyonuyla birleştirdim.