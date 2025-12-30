import random
from Metrix import path_metrics, total_cost
from Network import load_graph_and_demands

def ga_find_path(
    G, S, D,
    # NetworkX graph → Ağın tamamını temsil eder (node'lar ve edge'ler)
    # Her edge gecikme (delay), güvenilirlik (reliability),
    # bant genişliği (bandwidth) gibi QoS bilgileri içerir
    # Source → Yolun başlayacağı kaynak düğüm (başlangıç node'u)
    # Destination → Yolun biteceği hedef düğüm (varış node'u)
    
    w=(1/3, 1/3, 1/3),
    # Ağırlıklar (weights)
    # (delay_weight, reliability_weight, bandwidth_weight)
    # Fitness fonksiyonunda hangi kriterin ne kadar önemli olduğunu belirler
                
    demand=0.0,
    # Talep edilen minimum bant genişliği (QoS constraint)
    # Path üzerindeki tüm edge'lerin bandwidth >= demand olmalıdır
    # 0.0 ise bant genişliği kısıtı yoktur
                
    generations=50,
    # Genetik algoritmanın kaç nesil (iteration) çalışacağını belirtir
    # Nesil sayısı arttıkça çözüm kalitesi artabilir ama süre uzar
                
    pop_size=40,
    # Her nesildeki birey (path) sayısı
    # Popülasyon büyüdükçe çeşitlilik artar ama hesaplama maliyeti yükselir

    seed=0,
    # Rastgelelik için başlangıç değeri (random seed)
    # Aynı seed ile çalıştırıldığında aynı sonuçlar elde edilir
    # Deneylerin tekrarlanabilir olması için kullanılır
                
    elite_size=1,
    # Elitizm parametresi
    # Her nesilde en iyi kaç yolun doğrudan bir sonraki nesle
    # kopyalanacağını belirtir (kaybolmalarını önler)
               
    crossover_rate=0.9,
    # Çaprazlama (crossover) olasılığı
    # İki ebeveyn yolun genlerini karıştırarak yeni çocuk yol üretme ihtimali


    mutation_rate=0.30,
    # Mutasyon olasılığı
    # Yol üzerindeki bazı düğümlerin rastgele değiştirilme ihtimali
    # Genetik çeşitliliği korumak için kritik bir parametre
                
    max_init_tries=5000
    # Başlangıç popülasyonunu oluştururken
    # geçerli bir yol bulmak için yapılacak maksimum deneme sayısı
    # Sonsuz döngüye girmeyi engeller
                
):
    """
 GENETİK ALGORİTMA TABANLI QoS YOL BULMA YAKLAŞIMI – TASARIM AÇIKLAMASI
--------------------------------------------------------------------

Bu fonksiyon, ağ üzerinde kaynak (S) ve hedef (D) düğümleri arasında
en uygun yolu bulmak için **Genetik Algoritma (GA)** yaklaşımını kullanır.
Algoritmanın tasarımı bilinçli olarak klasik deterministik yöntemlerden
ayrıştırılmıştır.

 Shortest-path / Dijkstra / BFS / A* YOK
-----------------------------------------
- Algoritmada kesinlikle en kısa yol (shortest path) algoritmaları
  kullanılmamaktadır.
- Amaç, tek kriterli (mesafe) optimizasyon yerine;
  **çok kriterli (multi-objective) ve sezgisel (heuristic) bir çözüm**
  üretmektir.
- Bu sayede çözüm uzayı daha geniş tutulur ve QoS (Quality of Service)
  probleminin doğasına uygun bir yaklaşım elde edilir.

 Klon bireyleri doğrudan silme (remove_clones) YOK
---------------------------------------------------
- Popülasyondaki aynı yollar (klon bireyler) doğrudan silinmemektedir.
- Bunun yerine, genetik algoritmanın **doğal evrim mekanizmaları**
  kullanılarak klonların zamanla azalması hedeflenmiştir.
- Bu tercih, algoritmanın GA mantığına daha uygun ve daha akademik
  bir yaklaşımdır.

 Klonların kendiliğinden azalmasını sağlayan mekanizmalar:
-----------------------------------------------------------

1️⃣ Fitness Sharing (Paylaşımlı Uygunluk)
- Aynı veya çok benzer yollar popülasyonda arttıkça,
  bu yolların fitness değerleri bilinçli olarak düşürülür.
- Böylece tek bir yolun popülasyonu domine etmesi engellenir.
- Bu yöntem, genetik çeşitliliği (diversity) korumak için
  literatürde yaygın olarak kullanılan bir tekniktir.

2️⃣ Seçilim Baskısının Düşürülmesi
- Klasik elitist ve agresif seçim yöntemleri yerine,
  **rank-based roulette wheel** ve **stochastic selection**
  yaklaşımları kullanılmıştır.
- Amaç, sadece en iyi bireylerin değil,
  orta seviye çözümlerin de üreme şansı bulmasını sağlamaktır.
- Bu yaklaşım erken yakınsamayı (premature convergence) önler.

3️⃣ Üretim Basıncı ile Klon Azaltma (Hard Delete YOK)
- Yeni birey üretilirken, üst üste aynı çocuk yol oluşursa:
  → birey silinmez
  → ancak yeniden üretim denenir
- Bu yöntem, klonları zorla silmek yerine
  **üretim sürecini çeşitliliğe zorlayan yumuşak bir kontrol** sağlar.
- Genetik algoritmanın stokastik doğası korunur.

🎯 Sonuç:
---------
- Algoritma deterministik değildir; aynı parametrelerle bile
  benzer ama farklı çözümler üretebilir.
- Bu özellik, sezgisel optimizasyon algoritmaları için beklenen
  ve istenen bir davranıştır.
- Tasarım, klasik GA prensiplerine uygun, akademik olarak savunulabilir
  ve QoS tabanlı ağ yönlendirme problemleri için uygundur.
"""


    rnd = random.Random(seed)
# Rastgelelik için kontrollü bir random nesnesi oluşturulur
# Aynı seed değeri kullanıldığında:
# - Aynı başlangıç popülasyonu üretilir
# - Aynı seçim, çaprazlama ve mutasyon adımları gerçekleşir
# Bu sayede genetik algoritma deneyleri tekrarlanabilir (reproducible) olur


    # -------------------------------------------------
    # Geçerlilik Kontrolü
    # -------------------------------------------------
    def is_valid_path(p):
# Bu fonksiyon, genetik algoritma tarafından üretilen bir yolun
# ağ grafiği üzerinde gerçekten geçerli olup olmadığını kontrol eder.
# - Rastgelelik nedeniyle oluşabilecek hatalı / kopuk yolları elemek
# - Sadece fiziksel olarak mümkün yolların fitness hesaplamasına girmesini sağlamak
# - Bu fonksiyon bir optimizasyon yapmaz
# - Sadece "geçerli mi / değil mi" kontrolü uygular

        if not p or p[0] != S or p[-1] != D:
    #Temel yapı kontrolü
    # - Yol boş mu?
    # - Yol kaynak düğüm (S) ile mi başlıyor?
    # - Yol hedef düğüm (D) ile mi bitiyor?
    # Bu koşullardan biri sağlanmazsa yol geçersiz kabul edilir
            return False
        
        for i in range(len(p) - 1):
    #Kenar (edge) doğrulama kontrolü
    # Yol üzerindeki her iki ardışık düğüm arasında
    # grafikte gerçek bir bağlantı (edge) var mı diye kontrol edilir
    # p[i]     → mevcut düğüm
    # p[i + 1] → bir sonraki düğüm
    
            if not G.has_edge(p[i], p[i + 1]):
    # Eğer bu iki düğüm arasında edge yoksa:
    # - Bu geçiş ağda mümkün değildir
    # - Yol fiziksel olarak geçersizdir
        
                return False
            
        return True
    # Tüm kontroller geçildiyse
    # Yol:
    # - Kaynaktan başlıyor
    # - Hedefe ulaşıyor
    # - Sadece mevcut bağlantıları kullanıyor
    # Bu nedenle geçerli bir yol olarak kabul edilir
    

    # -------------------------------------------------
    # Yol Bant Genişliği (QoS)
    # -------------------------------------------------
    def path_min_bandwidth(p):
# Bu fonksiyon, verilen bir yol (path) üzerindeki
# EN DAR BOĞAZ (bottleneck) bant genişliğini hesaplar.
# Ağlarda bir yolun gerçek bant genişliği,
# o yol üzerindeki en düşük bant genişliğine sahip link tarafından belirlenir.
# Bu nedenle minimum (min) değer alınır.
# Eğer bir yol üzerindeki bant genişlikleri:
# [100, 80, 50, 120] ise
# bu yolun efektif bant genişliği = 50 olur

        if len(p) < 2:
   #Yol uzunluğu kontrolü
   # Eğer yol 2 düğümden kısa ise:
   # - Kenar (edge) yoktur
   # - Dolayısıyla bant genişliği hesaplanamaz
   # Bu durumda yol geçersiz kabul edilmez ama
   # bant genişliği 0.0 olarak döndürülür
   
            return 0.0
        return min(G.edges[p[i], p[i + 1]]["bandwidth"] for i in range(len(p) - 1))
    #Bant genişliği hesaplama (Bottleneck hesabı)
    # Yol üzerindeki her iki ardışık düğüm arasındaki
    # edge'lerin bant genişliği değerleri alınır
    # p[i]     → mevcut düğüm
    # p[i + 1] → bir sonraki düğüm
    # G.edges[u, v]["bandwidth"] ifadesi:
    # u ile v arasındaki bağlantının bant genişliğini temsil eder
    # min(...) kullanılarak:
    # - Yol üzerindeki en düşük bant genişliği bulunur
    # - Bu değer yolun QoS açısından taşıyabileceği maksimum kapasitedir
    

    # -------------------------------------------------
    # Random walk ile (Shortest Path yok!) yol üretimi
    # -------------------------------------------------
    def random_walk_to_target(start, target, visited=None, max_steps=None):

# Bu fonksiyonun amacı:
# - Genetik algoritma için "başlangıç popülasyonunu" (ilk bireyleri) üretmek
# - GA’nın ihtiyacı olan çeşitliliği (diversity) sağlamak
# - Deterministik en kısa yol algoritmalarını (Dijkstra, BFS, A*) kullanmadan
#   S -> D arasında "geçerli ve farklı" yollar üretmek

# Neden random walk?
# - GA’nın doğası stokastiktir (rastgele + evrimsel iyileştirme)
# - En kısa yol algoritması kullanırsak popülasyon aynı/benzer yollarla dolar,
#   bu da klon artışı ve erken yakınsamaya (premature convergence) yol açar.
# - Random walk başlangıçta çok farklı bireyler üretir; GA daha sonra
#   selection/crossover/mutation ile iyileştirir.

# Neden visited kullanıyoruz?
# - Random walk döngüye girebilir (A->B->A->B gibi)
# - visited seti, tekrar ziyaretleri azaltarak daha "anlamlı" yollar üretir
# - Bu "tam engelleme" değil, sadece tercih mekanizmasıdır (tıkandığında geri dönüş var)

# Neden demand filtresi var?
# - QoS routing probleminde "bandwidth >= demand" bir kısıt olabilir
# - Demand varsa, rastgele yürüyüşte bile kısıtı bozan edge'leri seçmeyerek
#   baştan daha uygulanabilir çözümler üretiriz
# - Bu, GA'nın fitness fonksiyonunun sürekli "geçersiz/cezalı birey" görmesini azaltır

# Neden max_steps var?
# - Graf büyükse random walk sonsuza yakın sürebilir veya döngüde kalabilir
# - max_steps, algoritmanın zamanını kontrol eder ve sonsuz döngüyü engeller
# - len(G)*3 seçimi: graf büyüdükçe izin verilen adım sayısı da büyüsün diye

# Neden tıkandığında (cand boş) geri dönüşe izin veriyoruz?
# - visited yüzünden hiç aday kalmayabilir (çıkmaz sokak)
# - Tamamen "visited dışı" şartı koyarsak çok sık None döner ve popülasyon dolmaz
# - Bu yüzden "yumuşak kural": önce ziyaret edilmemişleri dene, tıkanınca komşuların
#   tamamına izin ver (hard constraint değil, soft preference)

        if visited is None:
            visited = set()
    # visited parametresi dışarıdan verilmezse boş bir küme ile başlatılır.
    # Dışarıdan verilirse (örn: GA'da bazı düğümleri kilitlemek istersen) onu kullanabilir.
    
        if max_steps is None:
            max_steps = len(G) * 3


    # max_steps verilmezse default belirlenir.
    # len(G) = düğüm sayısı. *3 yapılması: "hedefe ulaşacak kadar dene ama sonsuz dolaşma" dengesi.
    
    # max_steps parametresi verilmezse varsayılan bir üst sınır belirlenir
    # len(G) = graf üzerindeki toplam düğüm (node) sayısı
    
    # Random walk deterministik değildir:
    # - Yanlış kollara sapabilir
    # - Çıkmaz sokaklara girebilir
    # - Geri dönüşler yapabilir
    
    #  Neden *1 değil?
    # - len(G) kadar adım, ideal ve hatasız bir yürüyüş varsayar
    # - Random seçimlerde bu çoğu zaman yeterli olmaz
    # - Hedefe ulaşmadan çok erken kesilme riski vardır
    
    #  Neden *100 değil?
    # - Gereksiz yere çok uzun random walk yapılır
    # - Hesaplama maliyeti artar
    # - Aynı düğümler arasında anlamsız dolaşma ihtimali yükselir
    # - GA için fayda sağlamayan zaman kaybına yol açar
    
    #  Neden *3?
    # - Random hatalara tolerans tanır
    # - Hedefe ulaşmak için yeterli esneklik sağlar
    # - Sonsuz veya anlamsız dolaşmayı engeller
    # - Pratikte iyi çalışan bir heuristic (sezgisel) denge noktasıdır
    
    # Ayrıca bu değer:
    # - Sabit değildir, graf boyutuna göre ölçeklenir
    # - Küçük graf → az adım, büyük graf → daha fazla adım
    # - GA’nın stokastik yapısına uygundur
    
    # Özet:
    # len(G) * 3 → "yeterince dene ama kilitlenme"
    

        current = start
        # Yürüyüşün (walk) başlangıç düğümü current olarak atanır.
        
        path = [current]
        # Path listesi, seçilen düğümlerin sırasını tutar (birey = path).
        
        visited = set(visited)
        # visited'i kopyalıyoruz:
    # - Dışarıdan gelen visited setini "yan etkiyle" bozmayalım diye.
    
        visited.add(current)
        # Başlangıç düğümünü ziyaret edildi olarak işaretleriz ki döngü olasılığı azalsın.
        

        steps = 0
        # Adım sayacı: max_steps sınırı için.
        
        while steps < max_steps:
        # max_steps aşılana kadar random walk devam eder.
            
            if current == target:
                return path
        # Eğer şu an hedef düğümdeysek, oluşturulan yolu döndürürüz.
        # Bu, random walk'in "başarılı" bitiş koşuludur.
        

            neighbors = list(G.neighbors(current))
        # current düğümünün komşularını alırız.
        # neighbors: grafın topolojisini kullanarak bir sonraki adımda nereye gidebileceğimizi belirler.
        

            
            cand = [n for n in neighbors if n not in visited]
        # ziyaret edilmemişleri tercih et (döngüleri azaltır)
        # Ziyaret edilmemiş komşuları tercih ediyoruz (cycle/döngü azaltma).
        # Bu bir "soft preference": sadece ilk seçim havuzunu oluşturur.

            
            if demand > 0.0:
                cand = [n for n in cand if G.edges[current, n]["bandwidth"] >= demand]
        # Eğer demand (minimum bant genişliği) kısıtı varsa,
        # adayları bandwidth >= demand şartına göre filtreleriz.
        # Böylece GA başlangıç bireylerinin: - QoS açısından daha uygulanabilir,
        # - fitness'ta daha az cezalı olmasını sağlarız.
       
        

            
            if not cand:
        # Eğer cand boş kaldıysa (tıkandıysak):
        # - visited yüzünden çıkmaz sokakta kalmış olabiliriz
        # - ya da demand filtresi yüzünden hiç uygun bağlantı kalmamış olabilir
        # Bu durumda "tamamen iptal" etmek yerine,
        # kilitlenmemek için komşuların tamamını denemeye izin veririz.
        # (Yani geri dönüş/backtracking benzeri yumuşak bir hareket)
        
                cand = neighbors[:]  # yine de bir çıkış dene
                # Tüm komşuları aday yaparak "çıkış" arıyoruz.
                
                if demand > 0.0:
                    cand = [n for n in cand if G.edges[current, n]["bandwidth"] >= demand]
            # demand varsa burada da filtremizi uygularız.
            # Çünkü demand bir QoS kısıtıysa bunu tamamen yok saymak istemeyiz.
            
                if not cand:
                    return None
            # Eğer hala hiç aday yoksa:
            # - Bu düğümden çıkış yok
            # - veya demand şartını sağlayan edge yok
            # Bu durumda yürüyüş başarısız olur ve None döndürür.
            # (GA bu bireyi atlayabilir / yeniden üretim deneyebilir)
            

            nxt = rnd.choice(cand)
        # Kandidatlar arasından rastgele bir sonraki düğümü seçiyoruz.
        # rnd.choice -> seed kontrollü random seçimi:
        # - Deney tekrarlanabilir
        # - Ama yöntem deterministik "en iyi" seçmez, rastgele seçer.
        
            path.append(nxt)
        # Seçilen düğümü path'e ekleriz (bireyin gen dizisi gibi düşün).
            
            visited.add(nxt)
        # Bu düğümü visited'a ekleriz:
        # - Bir sonraki adımlarda tekrar seçilme ihtimali azalır
        # - Döngüler azalır
        
            current = nxt
        # current'ı güncelleriz; yürüyüş bir sonraki düğümden devam eder.
            
            steps += 1
        # Bir adım tamamlandı → steps artırılır.
            

        return None
    # Eğer max_steps dolduysa ve hedefe ulaşamadıysak:
    # - Random walk bu denemede başarısız olmuştur None döndürür
    # Bu mekanizma GA için önemlidir:
    # - Başarısız üretim olursa başka bir deneme yapılabilir
    # - Sonsuz döngü engellenmiş olur
    



    def random_simple_path():
# -------------------------------------------------
# Basit rastgele yol üretimi (wrapper fonksiyon)
# -------------------------------------------------

# Bu fonksiyon, genetik algoritmanın farklı noktalarında
# (özellikle başlangıç popülasyonu üretimi sırasında)
# hızlı ve sade bir şekilde yol üretmek için kullanılır.

# random_simple_path:
# - S (source) düğümünden başlar
# - D (destination) düğümüne ulaşmaya çalışır
# - Shortest Path / Dijkstra / BFS / A* KULLANMAZ
# - Tamamen random_walk_to_target fonksiyonuna dayanır

        return random_walk_to_target(S, D, visited=None, max_steps=len(G) * 3)
# max_steps = len(G) * 3:
# - Graf büyüklüğüne göre dinamik adım sınırı
# - Sonsuz döngüleri engeller


    def repair_path(p):
# -------------------------------------------------
# PATH REPAIR (Shortest Path YOK!)
# -------------------------------------------------
# Bu fonksiyonun amacı:
# - Crossover veya mutation sonrası BOZULAN (invalid) path'leri
#   tamamen çöpe atmadan ONARMAK (repair)

# Neden repair mekanizması gerekli?
# - Genetik algoritmalarda çaprazlama ve mutasyon:
#   - Geçersiz kenarlar
#   - Kopuk yollar
#   - Hedefe ulaşmayan path'ler üretebilir

# Bu projede tercih edilen yaklaşım:
# Geçerli kısmı koru
# Kalan kısmı random-walk ile yeniden bağla

# Böylece:
# - Faydalı genetik bilgi (prefix path) kaybolmaz
# - Popülasyon çeşitliliği korunur
# - GA'nın evrimsel doğası bozulmaz


        if not p:
            return None
    #Boş path kontrolü
    # Eğer path None veya boşsa:
    # - Onarılacak bir şey yoktur
    # - Bu birey geçersiz kabul edilir
    
        if p[0] != S:
            return None
    #Kaynak düğüm kontrolü
    # Path mutlaka S düğümünden başlamalıdır
    # Aksi durumda yol tamamen hatalıdır ve onarılmaz
    

        
        fixed = [p[0]]
        visited = {p[0]}
    # Geçerli prefix (başlangıç zinciri) oluşturma
    # fixed: Path'in başından itibaren geçerli olan düğümleri tutar
    # - Bu kısım korunacak (genetik bilgi kaybı önlenir)
    # visited:
    # - Döngüleri azaltmak için
    # - Random walk sırasında tekrarları engellemek için kullanılır
    
        for i in range(1, len(p)):
    #Orijinal path üzerinde ilerleyerek
    # geçerli kenar zincirini koru
    
            u = fixed[-1]
            v = p[i]
        # u: fixed zincirinin son düğümü
        # v: orijinal path'teki sıradaki düğüm
        
            if v in visited:
                continue
        # Eğer bu düğüm daha önce ziyaret edildiyse:
        # - Döngü oluşur
        # - Bu düğüm atlanır
        
            if not G.has_edge(u, v):
                break
        # Eğer u -> v arasında edge yoksa:
        # - Buradan sonrası geçersizdir
        # - Prefix burada kesilir
        
            
            if demand > 0.0 and G.edges[u, v]["bandwidth"] < demand:
                break
        # Eğer demand (minimum bant genişliği) kısıtı varsa:
        # - Bu edge QoS şartını sağlamıyorsa
        # - Prefix burada kesilir
        # Bu kontrol sayesinde repair edilmiş path,
        # QoS açısından da anlamlı kalır
        
            fixed.append(v)
            visited.add(v)
        # Edge geçerliyse:
        # - Düğüm prefix'e eklenir
        
            if v == D:
                return fixed
        # Eğer hedef düğüme ulaşıldıysa:
        # - Repair tamamlanmıştır
        # - Prefix zaten tam bir yol haline gelmiştir
        

        
        tail = random_walk_to_target(fixed[-1], D, visited=visited, max_steps=len(G) * 3)
        #Tail = prefix’ten SONRA gelen ve hedefe (D) bağlanan YENİ üretilen yol parçası
        #Bozulan path’in “kalan kısmını” random walk ile yeniden ürettiğin bölüm

    # Prefix tamam ama D'ye ulaşmadıysak:
    # Kalan kısmı random walk ile bağlamaya çalışırız
    # Bu adım:
    # - Deterministik değildir
    # - Shortest path kullanmaz
    # - GA'nın stokastik doğasını korur
    
        if not tail:
            return None
    # Eğer random walk başarısız olduysa:
    # - Bu birey onarılamamıştır
    # - GA başka bireylerle devam edebilir
    

        
        merged = fixed + tail[1:]
        #Tail = prefix’ten SONRA gelen ve hedefe (D) bağlanan YENİ üretilen yol parçası
        #Bozulan path’in “kalan kısmını” random walk ile yeniden ürettiğin bölüm
        
    # tail'in ilk düğümü fixed[-1] zaten, tekrar etmeyelim
    # Prefix + tail birleştirme
    # tail[0] zaten fixed[-1] olduğu için
    # tekrar eklememek adına tail[1:] kullanılır
    
        return merged if is_valid_path(merged) else None
    # Son geçerlilik kontrolü
    # - Birleştirilen path gerçekten geçerli mi?
    # - Başlangıç S, bitiş D mi?
    # - Tüm kenarlar grafikte mevcut mu?
    # Eğer valid ise repaired path döndürülür,
    # değilse None döndürülür
    

    # -------------------------------------------------
    # Fitness (yumuşak ceza + length dengeleme)
    # -------------------------------------------------
    def base_fitness(p):
# Bu fonksiyon, genetik algoritmadaki HER bireyin (path)
# ne kadar "iyi" olduğunu sayısal bir değerle ifade eder.
# Düşük fitness değeri = daha iyi çözüm

# Tasarım felsefesi:
# - Sert (hard) kısıtlar yerine mümkün olduğunca YUMUŞAK CEZALAR kullanılır
# - Böylece GA:
#   * Çözüm uzayını daha iyi keşfeder (exploration)
#   * Erken yakınsamaya daha az girer
#   * Tamamen çöpe giden birey sayısı azalır
# Ayrıca:
# - Sadece en kısa yolu değil
# - QoS açısından dengeli yolları tercih eder

        
        d, rc, res = path_metrics(G, p)
    # path_metrics ve total_cost dışarıda tanımlı varsayılıyor
    # -------------------------------------------------
    # QoS metriklerinin hesaplanması
    # -------------------------------------------------
    # path_metrics fonksiyonu, verilen yol için:
    # d   → toplam gecikme (delay)
    # rc  → güvenilirlik bileşeni (reliability cost / reliability score)
    # res → kaynak kullanımı / residual kapasite vb. QoS metriği
    # değerlerini hesaplar
    
    # Bu fonksiyon dışarıda tanımlıdır ve
    # GA'dan bağımsız bir metrik hesaplama katmanıdır
    
        base_cost = total_cost(d, rc, res, w)
    # total_cost fonksiyonu:
    # - Yukarıda hesaplanan QoS metriklerini
    # - w ağırlıklarını kullanarak
    # tek bir skaler maliyet değerine indirger
    # Bu değer "ham maliyet"tir (henüz ceza eklenmemiştir)
    

        penalty = 0.0
    # -------------------------------------------------
    # Ceza (Penalty) teriminin başlatılması
    # -------------------------------------------------
    # Ceza, fitness değerine EKLENİR
    # (yani fitness büyüdükçe çözüm kötüleşir)
    

        if not is_valid_path(p):
            penalty += 1e6
    # -------------------------------------------------
    # Geçersiz yol için yüksek ceza (soft-hard hibrit)
    # -------------------------------------------------
    # Eğer yol:
    # - Ağ üzerinde fiziksel olarak mümkün değilse
    # - Kaynak-hedef şartını sağlamıyorsa
    # Çok büyük bir ceza eklenir
    # Not:
    # - Yol tamamen silinmez
    # - Ama seçilim şansı neredeyse sıfıra iner
    

        if demand > 0.0 and len(p) > 1 and is_valid_path(p):
    # -------------------------------------------------
    # Bandwidth (QoS) ihlali için yumuşak ceza
    # -------------------------------------------------
    # Eğer:
    # - Demand (minimum bant genişliği) tanımlıysa
    # - Yol en az bir edge içeriyorsa
    # - Yol geçerliyse
    # Yolun minimum bant genişliği hesaplanır
    
            min_bw = path_min_bandwidth(p)
            # Yol üzerindeki en dar boğaz (bottleneck) bant genişliği
            
            if min_bw < demand:
                penalty += (demand - min_bw) * 1000.0
        # Eğer bu değer demand'dan küçükse:
        # - Yol QoS şartını tam sağlamıyor demektir
        # - Ama tamamen geçersiz sayılmaz
        # İhlal miktarı kadar CEZA eklenir
        # (ne kadar ihlal, o kadar ceza)
        

        
        length_penalty = 0.01 * len(p)
    # shortest-path dominansını kıran ufak denge
    # -------------------------------------------------
    # Yol uzunluğu dengeleme cezası (Shortest-path baskısını kırma)
    # -------------------------------------------------
    # Eğer sadece base_cost kullanılsaydı:
    # - Çok kısa yollar aşırı avantajlı olurdu
    # - Popülasyon hızla shortest-path benzeri çözümlere çökerdi
    
    # Bu küçük ceza:
    # - Uzun yolları yumuşak şekilde dezavantajlı yapar
    # - Ama "en kısa yol zorunluluğu" oluşturmaz
    # Katsayı (0.01) bilinçli olarak küçük seçilmiştir
    

        return base_cost + penalty + length_penalty
    # -------------------------------------------------
    # Nihai fitness değeri
    # -------------------------------------------------
    # Toplam fitness:
    # - QoS tabanlı ham maliyet
    # - + geçersizlik cezaları
    # - + QoS ihlal cezaları
    # - + yol uzunluğu dengelemesi
    # GA bu değeri MINIMIZE etmeye çalışır
    



# -------------------------------------------------
# Fitness Sharing: aynı path çoğaldıkça "kendiliğinden" kötüleşsin
# (klon silmiyoruz, avantajlarını kırıyoruz)
# -------------------------------------------------


    def shared_fitness(p, counts, share_strength=0.15):
        # Bu fonksiyon, genetik algoritmada ortaya çıkan
        # AYNI veya BİREBİR AYNI (klon) yolların
        # popülasyonu domine etmesini ENGELLEMEK için kullanılır.
        
        # Temel fikir:
        # - Klon bireyleri doğrudan silmek (remove_clones) YOK
        # - Bunun yerine, aynı path çoğaldıkça fitness'ı
        #   "kendiliğinden" kötüleşsin
        
        # Bu yaklaşım:
        # - Klasik GA literatüründe "Fitness Sharing" olarak bilinir
        # - Genetik çeşitliliği (diversity) korumak için kullanılır
        # - Erken yakınsamayı (premature convergence) önler
        
        # Bu implementasyonda:
        # - Sharing işlemi path bazında yapılır (node dizisi)
        # - Aynı diziyi taşıyan bireyler klon kabul edilir
        
        f = base_fitness(p)
    # -------------------------------------------------
    # Temel fitness hesaplama
    # -------------------------------------------------
    # Önce path'in normal fitness değeri hesaplanır
    # (QoS + ceza + uzunluk dengesi)
    
        c = counts.get(tuple(p), 1)
    # minimization: c arttıkça f büyüsün
    # -------------------------------------------------
    # Klon sayısının bulunması
    # -------------------------------------------------
    # counts:
    # - Popülasyondaki path'lerin kaç kez tekrarlandığını tutan sözlük
    # - Anahtar: tuple(p)  → path'in hashlenebilir hali
    # - Değer: aynı path'ten kaç tane olduğu
    # Eğer path sözlükte yoksa:
    # - Varsayılan olarak 1 kabul edilir (tekil birey)
    
        return f * (1.0 + share_strength * (c - 1))
    # -------------------------------------------------
    # Fitness sharing uygulanması
    # -------------------------------------------------
    # Bu problem MINIMIZATION problemidir:
    # - Fitness ne kadar küçükse çözüm o kadar iyidir
    
    # Formül:
    # shared_fitness = f * (1 + share_strength * (c - 1))
    
    # Anlamı:
    # - c = 1  → (tekil birey) → fitness değişmez
    # - c > 1  → (klonlar var) → fitness çarpanla büyür
    
    # share_strength:
    # - Klonlara uygulanacak cezanın şiddetini belirler
    # - Küçük seçilmiştir (0.15):
    #   * Klonları tamamen öldürmez
    #   * Ama avantajlarını kırar
    
    # Böylece:
    # - Popülasyon tek bir path'e çökmez
    # - Farklı çözümler hayatta kalma şansı bulur
    

    # -------------------------------------------------
    # Rank-based Roulette (seçilim baskısı düşük, diversity daha iyi)
    # -------------------------------------------------
    def rank_roulette_selection(pop, fit_vals):
# Bu fonksiyon, genetik algoritmada "ebeveyn seçimi" için kullanılır.
# Neden klasik roulette wheel DEĞİL?
# - Klasik roulette doğrudan fitness değerine bağlıdır
# - En iyi bireyler aşırı baskın olur
# - Popülasyon hızlıca klonlaşır
# - Erken yakınsama (premature convergence) oluşur

# Rank-based yaklaşımın avantajı:
# - Fitness değerinin mutlak büyüklüğü değil
# - Popülasyon içindeki GÖRECELİ SIRALAMA önemlidir
# - Seçilim baskısı daha yumuşaktır
# - Orta seviye çözümler de yaşama şansı bulur

        ranked_idx = sorted(range(len(pop)), key=lambda i: fit_vals[i])
        n = len(pop)
        # Popülasyon büyüklüğü

    # -------------------------------------------------
    # Fitness'a göre sıralama (küçük daha iyi)
    # -------------------------------------------------
    # ranked_idx:
    # - popülasyondaki bireylerin indekslerini tutar
    # - fitness değerine göre artan sırayla dizilir
    
    # Yani:
    # ranked_idx[0] → en iyi bireyin indeksi
    # ranked_idx[-1] → en kötü bireyin indeksi
    
        scores = [0.0] * n
   # -------------------------------------------------
   # Rank tabanlı skor atama
   # -------------------------------------------------
   # scores dizisi:
   # - Her bireyin seçilme ağırlığını tutar
   # - Başlangıçta tüm skorlar 0
   
        for rank, i in enumerate(ranked_idx):
    # En iyi bireye en yüksek skor,
    # en kötü bireye en düşük skor verilir
    
            scores[i] = (n - rank)
        # rank = 0 → en iyi birey
        # score = n
        # rank = n-1 → en kötü birey
        # score = 1
        

        scores = [s ** 0.7 for s in scores]
    # -------------------------------------------------
    # Seçilim baskısını yumuşatma
    # -------------------------------------------------
    # scores ** 0.7:
    # - En iyi bireyin avantajını kırar
    # - Kötü bireyleri tamamen yok etmez
    
    # Üs < 1 olduğu için:
    # - Skorlar birbirine yaklaşır
    # - Seçilim baskısı azalır
    
    # Bu, diversity (çeşitlilik) için kritiktir
    

        total = sum(scores)
    # -------------------------------------------------
    # Roulette wheel seçimi
    # -------------------------------------------------
    # Toplam skor hesaplanır
    
        pick = rnd.uniform(0, total)
# -------------------------------------------------
# ROULETTE WHEEL SEÇİMİNİN ASIL ÇALIŞTIĞI KISIM
# -------------------------------------------------

# [0, total] aralığında rastgele bir eşik (threshold) seçilir
# total = tüm bireylerin seçim skorlarının toplamıdır

# Bu eşik:
# - Hangi bireyin seçileceğini belirler
# - Daha yüksek skora sahip bireylerin
#   bu eşiği "geçme" ihtimali daha yüksektir

# rnd.uniform kullanılması:
# - Deterministik seçim YOK
# - Aynı fitness sıralamasında bile
#   farklı bireyler seçilebilir
# - GA'nın stokastik (rastgele) doğası korunur
    
        acc = 0.0
# Kümülatif skor (accumulator)
# Bu değişken, skorları soldan sağa toplayarak ilerler
#Bu değişkenin sebebi, "Rulet Tekerleği Seçimi" (Roulette Wheel Selection)
#mantığını kodlayabilmek için sınır çizgilerini belirlemektir.
#Bunu en basit haliyle "Uç Uca Ekleme" mantığıyla açıklayabiliriz.

        for p, s in zip(pop, scores):
# pop  → bireyler (path'ler)
# scores → her bireyin seçilme ağırlığı

# zip(pop, scores):
# - Her birey kendi skoruyla birlikte ele alınır

            acc += s
        # Her adımda mevcut bireyin skoru kümülatif toplama eklenir
        
            if acc >= pick:
                return p
    # Eğer kümülatif skor, rastgele seçilen eşiği geçtiyse:
    # - Bu birey "roulette wheel" üzerinde seçilmiş olur
    
    # Görsel olarak:
    # [---- p1 ----|---- p2 ----|---- p3 ----| ... ]
    #              ↑
    #           pick buraya düştü → p2 seçildi
    
    # Bu mekanizma sayesinde:
    # - Skoru yüksek bireylerin aralığı geniştir
    # - Seçilme ihtimalleri daha fazladır
    # - Ama skoru düşük bireylerin ihtimali sıfır değildir
        
        return pop[-1]
# -------------------------------------------------
# FALLBACK (GÜVENLİ DÖNÜŞ)
# -------------------------------------------------
# Teorik olarak bu noktaya gelinmemelidir
# Ancak:
# - Floating point hassasiyet hataları
# - Çok küçük skorlar
# - Toplamın tam örtüşmemesi
# gibi nadir durumlar olabilir

# Bu yüzden:
# - Fonksiyon mutlaka bir birey döndürsün diye
# - Son birey güvenli fallback olarak seçilir

# Bu satır:
# - GA mantığını bozmaz
# - Programın çökmesini önler
    

    # -------------------------------------------------
    # Crossover – Path-Aware + Fallback
    # -------------------------------------------------
    def crossover_1point(p1, p2):
# Bu bölüm, genetik algoritmada iki ebeveyn path'ten
# yeni bir çocuk path üretmek için kullanılır.

# Tasarım felsefesi:
# - Klasik one-point crossover tek başına yeterli değil
# - Path problemlerinde, düğüm dizilerinin anlamı vardır
# - Rastgele kesme çoğu zaman kopuk / geçersiz path üretir

# Bu yüzden çaprazlama (crossover) iki aşamalı tasarlanmıştır:
# 1️⃣ Önce PATH-AWARE CROSSOVER denenir:
#    - İki ebeveyn path üzerinde ortak bir düğüm (intersection) aranır
#    - Ortak düğüm, yapısal olarak anlamlı bir birleşme noktasıdır
#    - Bu noktadan yapılan birleşim:
#      * Kopuk path üretme ihtimalini azaltır
#      * Daha tutarlı ve geçerli çocuklar üretir
#    - Böylece genetik bilgi "anlamlı şekilde" aktarılır

# 2️⃣ Eğer PATH-AWARE CROSSOVER mümkün değilse:
#    - Yani ebeveynler arasında ortak bir düğüm yoksa
#    - Anlamlı bir birleşme noktası da yoktur
#    - Bu durumda algoritma kilitlenmemek için
#      ONE-POINT CROSSOVER'a geri düşer (fallback)

# Fallback kullanımı:
# - Algoritmanın her zaman çocuk üretebilmesini garanti eder
# - Çeşitliliği korur (tamamen aynı yapılar üretilmez)
# - Ama ana hedef olan "anlamlı birleşim"den vazgeçilmez

# Özetle:
# - Öncelik: Yapısal olarak tutarlı crossover
# - Alternatif: Basit ama güvenli crossover
# - Deterministik shortest-path benzeri birleşimler KULLANILMAZ

# Böylece:
# - Genetik çeşitlilik korunur
# - Geçersiz çocuk sayısı azalır
# - Shortest-path benzeri deterministik yapı oluşmaz

# -------------------------------------------------
# KLASİK ONE-POINT CROSSOVER (Fallback)
# -------------------------------------------------
# Bu fonksiyon, iki ebeveyn path arasında
# klasik tek nokta çaprazlama uygular.

# Ne zaman kullanılır?
# - Path-aware crossover için ortak düğüm yoksa
# - Yani ebeveynlerin yapısal bir birleşme noktası yoksa

        if len(p1) < 2 or len(p2) < 2:
            return p1[:]
    # Eğer ebeveynlerden biri çok kısa ise:
    # - Anlamlı bir kesme noktası yoktur
    # - p1'in kopyası döndürülür
    
        cut = rnd.randint(1, min(len(p1), len(p2)) - 1)
        # Kesme noktası:
        # - 1 ile min(len(p1), len(p2)) - 1 arasında rastgele seçilir
        # - Başlangıç ve bitiş düğümleri korunur
        # -------------------------------------------------
# ONE-POINT CROSSOVER KESME NOKTASI SEÇİMİNİN MANTIĞI
# -------------------------------------------------

# cut = rnd.randint(1, min(len(p1), len(p2)) - 1)
# Bu kesme aralığı RASTGELE değil, PATH tabanlı GA için
# BİLİNÇLİ ve AKADEMİK olarak seçilmiştir.

# =================================================
#  Neden 0'dan BAŞLAMIYOR?
# =================================================
# Eğer cut = 0 olsaydı:
# - p1[:0] → boş liste
# - child = tamamen p2 olurdu

# Sonuç:
# - Gerçek bir çaprazlama gerçekleşmez
# - Genetik bilgi SADECE tek ebeveynden gelir
# - Crossover operatörü anlamsızlaşır

# GA mantığında crossover:
# → İKİ ebeveynden de genetik bilgi almalıdır
#
# Bu yüzden cut >= 1 şartı zorunludur.

# =================================================
# Neden 1'DEN BAŞLIYOR?
# =================================================
# cut = 1 olduğunda:
# - p1'in ilk düğümü (S - source) korunur

# Bu çok kritiktir çünkü:
# - Tüm geçerli path'ler S'den başlamak zorundadır
# - Eğer başlangıç düğümü bozulursa:
#   * Yol geçersiz olur
#   * Repair ihtiyacı artar
#   * GA gereksiz yere ceza yer

# Bu seçimle:
# - Çocuk path mutlaka doğru kaynaktan başlar
# - Fiziksel geçerlilik ihtimali artar

# =================================================
# Neden min(len(p1), len(p2)) KULLANILIYOR?
# =================================================
# p1 ve p2 farklı uzunluklarda olabilir

# Eğer sadece len(p1) kullanılsaydı:
# - p2 için geçersiz index riski oluşurdu
# - veya p2'den anlamsız genler alınabilirdi

# min(...) kullanılarak:
# - Kesme noktası HER İKİ ebeveyn için de güvenli olur
# - Index hataları önlenir
# - Path yapısı korunur

# =================================================
# Neden -1 YAPILIYOR?
# =================================================
# Son düğüm (D - destination) korunmak istenir

# Eğer cut son indexi kapsasaydı:
# - p2'den hiçbir gen alınmazdı
# - child, p1'in birebir kopyası olurdu

# Bu durumda:
# - Çeşitlilik artmaz
# - Klon bireyler çoğalır
# - GA erken yakınsamaya girer

# Bu yüzden:
# - cut < len(path) - 1 şartı konur

# =================================================
# PATH TABANLI PROBLEMLERDE ÖZEL DURUM
# =================================================
# Bu problem klasik "bit string GA" değildir.
# Burada genler:
# - Rastgele sayılar değil
# - Ağ düğümleri (node'lar)dır

# Yanlış kesme noktası:
# - Kopuk edge'ler
# - Geçersiz yollar
# - Aşırı repair ihtiyacı
# doğurur

# Bu kesme aralığı:
# - Başlangıcı (S) korur
# - Bitişi (D) korur
# - Ortada anlamlı genetik karışım sağlar

# =================================================
# GA VE AKADEMİK SONUÇ
# =================================================
# Bu kesme noktası seçimi sayesinde:
# - Crossover gerçekten "genetik karışım" yapar
# - Çocuk path'ler daha sık geçerli olur
# - Klon üretimi azalır
# - Repair mekanizmasına aşırı yük binmez

# Bu yaklaşım:
# - Klasik GA prensipleriyle uyumludur
# - Path-based optimization problemleri için uygundur
# - Shortest-path benzeri deterministik yapı üretmez

# Kesme noktası, başlangıç ve hedef düğümleri koruyacak,
# iki ebeveynden de anlamlı genetik bilgi alacak
# ve path geçerliliğini mümkün olduğunca bozmayacak
# şekilde bilinçli olarak sınırlandırılmıştır.

        
        return p1[:cut] + p2[cut:]
    # Çocuk path:
    # - p1'den kesme noktasına kadar
    # - p2'den kesme noktasından sonrası
    # Bu yöntem:
    # - Basit
    # - Stokastik
    # - Ama bazen kopuk path üretebilir (bu yüzden fallback'tir)
    

    def crossover_path_aware(p1, p2):
# -------------------------------------------------
# PATH-AWARE CROSSOVER (Öncelikli yöntem)
# -------------------------------------------------
# Bu fonksiyon, path problemlerine ÖZEL bir çaprazlama uygular.

# Temel fikir:
# - İki ebeveyn path üzerinde ORTAK bir düğüm (intersection) varsa
# - Bu düğüm doğal bir birleşme noktasıdır
# - Kopukluk ihtimali azalır

# Neden başlangıç ve bitiş düğümleri hariç tutulur?
# - S ve D tüm path'lerde ortaktır
# - Anlamlı genetik çeşitlilik üretmez

        common = list(set(p1[1:-1]) & set(p2[1:-1]))
    # Ortak düğümlerin bulunması
    # p1[1:-1], p2[1:-1]:
    # - Başlangıç (S) ve bitiş (D) düğümleri hariç tutulur
    # set & set:
    # - İki path'in kesişim kümesi
    
        if not common:
            return crossover_1point(p1, p2)
    # Ortak düğüm yoksa:
    # - Anlamlı bir birleşme noktası yoktur
    # - Path-aware crossover yapılamaz
    # - Fallback olarak one-point crossover kullanılır
    

        mid = rnd.choice(common)
    # Ortak düğümler arasından rastgele bir orta düğüm seçilir
    # Bu düğüm, crossover noktası gibi davranır
    
        i1 = p1.index(mid)
        i2 = p2.index(mid)
    # Seçilen düğümün ebeveynlerdeki indeksleri bulunur
    # Bu indeksler, path'in hangi noktadan bölüneceğini belirler
    
        child = p1[:i1] + p2[i2:]
    # Çocuk path oluşturma
    # p1[:i1]:
    # - Başlangıçtan ortak düğümden ÖNCEKİ kısım
    # p2[i2:]:
    # - Ortak düğüm ve sonrası
    # Bu birleşim:
    # - Yapısal olarak daha tutarlı
    # - Genelde geçerli edge zinciri üretme ihtimali daha yüksek
    
        return child
    # Oluşturulan çocuk path döndürülür
    # (Gerekirse repair mekanizması daha sonra devreye girer)
    

    # -------------------------------------------------
    # Mutasyon
    # -------------------------------------------------
    def mutate_random_gene(p):
# -------------------------------------------------
# Bu bölüm, genetik algoritmanın en kritik parçalarından biridir.

# Mutasyonun temel amacı:
# - Popülasyona YENİ GENETİK BİLGİ eklemek
# - Klonlaşmayı engellemek
# - Yerel minimumlara sıkışmayı önlemek

# Bu implementasyonda:
# - Tek tip mutasyon YOK
# - İki farklı mutasyon stratejisi vardır
# - Her birey için rastgele biri seçilir

# Bu yaklaşım:
# - Klasik GA mantığına uygundur
# - Exploration (keşif) yeteneğini artırır
# - QoS routing problemine daha uygundur


# -------------------------------------------------
# MUTASYON TÜRÜ 1: RANDOM GENE MUTATION
# -------------------------------------------------
# Bu mutasyon:
# - Path içindeki TEK bir düğümü değiştirir
# - Küçük ama etkili bir genetik oynama yapar

# Amaç:
# - Yolun genel yapısını bozmadan
# - Yerel varyasyon oluşturmak

        if len(p) < 3:
            return p[:]
    # Eğer path çok kısaysa:
    # - Başlangıç (S) ve bitiş (D) dışında
    #   değiştirilecek anlamlı bir gen yoktur
    
        child = p[:]
        # Orijinal path kopyalanır (yan etkiyi önlemek için)
        
        idx = rnd.randint(1, len(child) - 2)
# -------------------------------------------------
# MUTASYONA UĞRAYACAK GENİN (DÜĞÜMÜN) SEÇİLMESİ
# -------------------------------------------------
# idx:
# - Path (child) içindeki HANGİ düğümün değiştirileceğini belirler

# Neden 1 ile len(child) - 2 arası?

# Neden 0 seçilmiyor?
# - idx = 0 olsaydı:
#   * Başlangıç düğümü (S - source) değişirdi
#   * Path kaynaktan başlamaz hale gelirdi
#   * Yol fiziksel olarak geçersiz olurdu

# Neden son indeks seçilmiyor?
# - idx = len(child) - 1 olsaydı:
#   * Hedef düğüm (D - destination) değişirdi
#   * Yol hedefe ulaşmaz hale gelirdi

# Sonuç:
# - S ve D genleri SABİT tutulur
# - Sadece ara düğümler (intermediate nodes) mutasyona uğrar
# - Path'in temel yapısı korunur

# Bu yaklaşım:
# - Path tabanlı GA'lar için standart ve gereklidir
# - Aksi halde her mutasyon geçersiz birey üretir

# -------------------------------------------------
# NEDEN RASTGELE SEÇİLİYOR?
# -------------------------------------------------
# rnd.randint:
# - Mutasyonun deterministik olmasını engeller
# - Aynı birey her seferinde farklı şekilde mutasyona uğrayabilir
# - GA'nın stokastik doğasına uygundur

# Bu, exploration (keşif) yeteneğini artırır
    
        prev = child[idx - 1]
# -------------------------------------------------
# MUTASYON NOKTASINDAN ÖNCEKİ DÜĞÜMÜN BELİRLENMESİ
# -------------------------------------------------
# prev:
# - Mutasyona uğrayacak düğümden BİR ÖNCEKİ düğümdür

# Neden prev kullanılıyor?
# - Path bir düğüm dizisidir
# - child[idx] düğümü değiştirilecekse:
#   * Yeni düğüm, child[idx - 1]'e KOMŞU olmak zorundadır
#   * Aksi halde u -> v arasında edge yoksa path kopar

# Yani:
# - Mutasyon "rastgele düğüm atama" değildir
# - Yapısal tutarlılık korunur

# Bu yaklaşım:
# - Fiziksel olarak mümkün olmayan yolların üretilmesini azaltır
# - Repair mekanizmasına olan ihtiyacı düşürür

    

        neighbors = list(G.neighbors(prev))
# -------------------------------------------------
# GEÇERLİ ADAY GENLERİN (KOMŞULARIN) BELİRLENMESİ
# -------------------------------------------------
# G.neighbors(prev):
# - Graf üzerinde prev düğümüne DOĞRUDAN bağlı düğümleri döndürür

# Bu liste:
# - Mutasyon sonucu child[idx] için seçilebilecek
#   TÜM FİZİKSEL OLARAK GEÇERLİ düğümleri içerir

# Neden sadece komşular?
# - Path problemlerinde genler bağımsız değildir
# - Rastgele bir düğüm seçilirse:
#   * prev → new_node arasında edge olmayabilir
#   * Path kopar ve geçersiz olur

# Bu nedenle mutasyon:
# - Rastgele ama KONTROLLÜ yapılır
# - Graf topolojisine saygılıdır

# GA açısından sonuç:
# - Mutasyon çeşitlilik sağlar
# - Ama path yapısını tamamen bozmaz
# - Geçerli birey üretme olasılığı yüksek kalır
        
        if demand > 0.0:
            neighbors = [n for n in neighbors if G.edges[prev, n]["bandwidth"] >= demand]
    # Eğer bandwidth demand kısıtı varsa:
    # - QoS ihlali oluşturmamak için
    # - Komşular filtrelenir
    
        if neighbors:
            child[idx] = rnd.choice(neighbors)
    # Eğer uygun komşu varsa:
    # - Yeni gen bu komşular arasından rastgele seçilir
    # - Böylece path yapısal olarak kopmaz
    
        return child
    # Mutasyona uğramış child path döndürülür
    

    def mutate_segment_reset(p):
# -------------------------------------------------
# MUTASYON TÜRÜ 2: SEGMENT RESET MUTATION
# -------------------------------------------------
# Bu mutasyon:
# - Path'in bir kısmını tamamen sıfırlar
# - Daha büyük yapısal değişiklik oluşturur
#
# Amaç:
# - Yerel minimumdan çıkmak
# - Path'in ikinci yarısında yeni bölgeler keşfetmek
#
# Bu, klasik GA'daki "strong mutation" benzeri bir yaklaşımdır

        if len(p) < 4:
            return p[:]
    # Path çok kısa ise:
    # - Segment reset anlamlı değildir
    
        cut = rnd.randint(1, len(p) - 2)
    # Kesme noktası seçilir
    # - 1 ile len-2 arası
    # - S korunur, D resetlenecek kısımda kalır
    
        head = p[:cut]
        # Path'in baş kısmı (prefix) korunur
        
        tail = random_walk_to_target(head[-1], D, visited=set(head), max_steps=len(G) * 3)
    # Kalan kısmı:
    # - head'in son düğümünden başlayarak
    # - random walk ile D'ye bağlamaya çalışırız
    
    # visited = set(head):
    # - Prefix düğümleri tekrar edilmesin
    # - Döngüler azalsın
    
        if not tail:
            return p[:]
    # Eğer random walk başarısız olursa:
    # - Mutasyon başarısız sayılır
    # - Orijinal path korunur
    
        child = head + tail[1:]
    # Prefix + yeni oluşturulan tail birleştirilir
    # tail[0] zaten head[-1] olduğu için tekrar eklenmez
    
        return child
    # Yeni oluşturulan path döndürülür
    

    def mutate(p):
# -------------------------------------------------
# MUTASYON SEÇİCİSİ (MIXED STRATEGY)
# -------------------------------------------------
# Bu fonksiyon:
# - Hangi mutasyon türünün uygulanacağını belirler
#
# %50 ihtimalle:
# - Random gene mutation (küçük değişim)
# %50 ihtimalle:
# - Segment reset mutation (büyük değişim)
#
# Bu denge:
# - Exploration (keşif)
# - Exploitation (iyileştirme)
# arasında denge kurar

        if rnd.random() < 0.5:
            return mutate_random_gene(p)
# Rastgele bir sayı üretilir (0.0 ile 1.0 arasında)
# rnd.random() < 0.5 koşulu:
# - %50 olasılıkla TRUE olur
# - %50 olasılıkla FALSE olur

# Bu yapı sayesinde:
# - Mutasyon tipi deterministik olmaz
# - Her birey her seferinde aynı şekilde mutasyona uğramaz
# - Genetik çeşitlilik (diversity) artar

# %50 ihtimalle:
# - Küçük ölçekli bir mutasyon uygulanır
# - Path'in genel yapısı korunur
# - Yerel iyileştirme (exploitation) sağlanır

        else:
            return mutate_segment_reset(p)
# Kalan %50 ihtimalle:
# - Daha agresif bir mutasyon uygulanır
# - Path'in bir bölümü tamamen sıfırlanır
# - Yeni bölgeler keşfedilir (exploration)
# - Yerel minimumdan çıkma ihtimali artar


# -------------------------------------------------
# Başlangıç Popülasyonu
# -------------------------------------------------
# Bu bölüm, genetik algoritmanın ilk neslini (initial population) oluşturur.

# Genetik algoritmada başlangıç popülasyonu:
# - Çözüm uzayının hangi bölgesinden başlanacağını belirler
# - Algoritmanın başarısını doğrudan etkiler

# Bu implementasyonda:
# - Deterministik shortest-path algoritmaları KULLANILMAZ
# - Tamamen random walk tabanlı yol üretimi yapılır
# - Ama sadece GEÇERLİ yollar popülasyona alınır

# Amaç:
# - Yüksek çeşitlilik (diversity)
# - Fiziksel olarak mümkün bireyler
# - QoS açısından en azından uygulanabilir çözümler

    population = []
    # Popülasyonu tutacak liste
    
    tries = 0
    # Kaç deneme yapıldığını takip eder
    # (sonsuz döngüyü önlemek için)

    while len(population) < pop_size and tries < max_init_tries:
    # Popülasyon dolana kadar VE
    # maksimum deneme sayısı aşılmadığı sürece

        tries += 1
        # Her denemede sayaç artırılır
        
        p = random_simple_path()
        # Random walk tabanlı bir yol üretilir
        # (random_simple_path → S'den D'ye rastgele yürüyüş)
    
        if p and is_valid_path(p):
            population.append(p)
    # Üretilen yol:
    # - None değilse
    # - Ağ üzerinde fiziksel olarak geçerliyse
    # popülasyona eklenir
    
    # Bu kontrol sayesinde:
    # - Geçersiz bireyler baştan elenir
    # - Fitness fonksiyonu "çöp" bireylerle uğraşmaz
    

    if not population:
        return [S, D], (float("inf"), float("inf"), float("inf"))
# -------------------------------------------------
# Güvenlik kontrolü (fallback)
# -------------------------------------------------
# Eğer tüm denemelere rağmen
# TEK BİR geçerli yol bile üretilememişse:

# - GA'nın çalışması anlamsız hale gelir
# - Bu durumda algoritma çökmek yerine
#   güvenli bir fallback döndürür

# [S, D]:
# - En basit olası yol
# - Grafikte edge varsa geçerli,
#   yoksa fitness ile cezalandırılacaktır

# (inf, inf, inf):
# - QoS metriklerinin anlamsız olduğunu belirtir
# - Üst katman kodun durumu fark etmesini sağlar


# -------------------------------------------------
# Evrim Döngüsü
# -------------------------------------------------
# Bu bölüm, genetik algoritmanın "evrim" sürecini yönetir.

# Burada gerçekleşenler:
# - Fitness hesaplama
# - Seçilim (selection)
# - Çaprazlama (crossover)
# - Mutasyon (mutation)
# - Onarım (repair)
# - Çeşitlilik enjeksiyonu

# Tüm süreç, klasik GA akışına uygundur
# ancak erken yakınsamayı önlemek için
# yumuşak ve çeşitlilik dostu mekanizmalar içerir


    elite_size = max(0, min(elite_size, pop_size))
# -------------------------------------------------
# ELİTİZM (ELITISM) – EN İYİ BİREYLERİN KORUNMASI
# -------------------------------------------------

# elite_size:
# - Her nesilde EN İYİ kaç bireyin (path'in)
#   doğrudan bir sonraki nesle kopyalanacağını belirler

# Neden elitizm kullanıyoruz?
# - Genetik algoritmalarda:
#   * Seçilim
#   * Çaprazlama
#   * Mutasyon
#   işlemleri tamamen stokastiktir

# - Bu rastgelelik yüzünden:
#   * Çok iyi bir çözüm
#   * Bir sonraki nesilde KAYBOLABİLİR

# Elitizm sayesinde:
# - En iyi çözümler "sigortalanır"
# - Fitness değeri iyi olan path'ler yok olmaz
# - GA'nın çözüm kalitesi nesiller boyunca GERİYE GİTMEZ

# -------------------------------------------------
# Neden max(0, min(elite_size, pop_size))?
# -------------------------------------------------
# Bu satır bir GÜVENLİK ÖNLEMİDİR:

# elite_size negatif olamaz
#    → max(0, ...)
#
# elite_size, pop_size'dan büyük olamaz
#    → min(elite_size, pop_size)

# Böylece:
# - Parametre hataları GA'nın çökmesine yol açmaz
# - Elitizm mantıklı sınırlar içinde kalır

# -------------------------------------------------
# GA açısından elitizmin etkisi:
# -------------------------------------------------
# + Çözüm kalitesi korunur
# + En iyi bireyler kaybolmaz
# - Aşırı elitizm olursa:
#   * Çeşitlilik azalır
#   * Popülasyon klonlaşır

# Bu yüzden elite_size küçük tutulmuştur (genelde 1 veya 2)


    inject_count = max(1, int(pop_size * 0.10))
# -------------------------------------------------
# ÇEŞİTLİLİK ENJEKSİYONU (RANDOM IMMIGRANTS)
# -------------------------------------------------

# inject_count:
# - Her nesilde popülasyona DIŞARIDAN eklenecek
#   yeni rastgele birey (path) sayısını belirtir

# Bu bireyler:
# - Mevcut popülasyondan türetilmez
# - Tamamen random walk ile üretilir
# - GA'nın mevcut yönelimine BAĞLI DEĞİLDİR

# -------------------------------------------------
# Neden enjeksiyon (injection) yapıyoruz?
# -------------------------------------------------
# Genetik algoritmalarda en büyük problemlerden biri:
# → Erken yakınsama (premature convergence)

# Bu ne demek?
# - Popülasyon çok erken tek bir çözüm etrafında toplanır
# - Yeni bölgeler keşfedilmez
# - Daha iyi çözümler kaçırılır
#
# Random immigrants (enjeksiyon) sayesinde:
# - Popülasyona her nesilde "taze genetik bilgi" girer
# - Daha önce hiç görülmemiş path'ler denenir
# - GA'nın keşif (exploration) yeteneği artar

# -------------------------------------------------
# Neden popülasyonun %10'u?
# -------------------------------------------------
# %10:
# - Literatürde sık kullanılan
# - Pratikte iyi sonuç veren
# - Dengeli bir orandır

# Daha az olsaydı:
# - Etkisi zayıf olurdu

# Daha fazla olsaydı:
# - Evrimsel öğrenme bozulurdu
# - GA rastgele aramaya yaklaşırdı

# -------------------------------------------------
# Neden max(1, ...) kullanılıyor?
# -------------------------------------------------
# Küçük popülasyonlarda:
# - int(pop_size * 0.10) = 0 olabilir

# max(1, ...) ile:
# - Her nesilde EN AZ 1 yeni birey
#   popülasyona mutlaka eklenir

# Bu:
# - GA'nın tamamen kilitlenmesini önler
# - Uzun koşularda keşfi garanti eder

# -------------------------------------------------
# ELİTİZM + ENJEKSİYON BİRLİKTE NE SAĞLAR?
# -------------------------------------------------

# Elitizm:
# - İyi çözümleri KORUR (exploitation)

# Enjeksiyon:
# - Yeni çözümler KEŞFETTİRİR (exploration)

# Bu ikisi birlikte:
# - Klasik GA'nın temel dengesini kurar
# - Hem kaliteyi hem çeşitliliği aynı anda yönetir

# Elitizm ile en iyi çözümler korunurken,
# rastgele birey enjeksiyonu ile popülasyonun
# genetik çeşitliliği her nesilde taze tutulur.


    for generation in range(generations):
# -------------------------------------------------
# NESİL DÖNGÜSÜ
# -------------------------------------------------
# GA, belirlenen nesil sayısı kadar çalışır
#amaç popülasyondaki kopyaları yakalamak

        counts = {}
        for p in population:
            k = tuple(p)
            counts[k] = counts.get(k, 0) + 1
    # -------------------------------------------------
    # Klon sayılarının hesaplanması
    # -------------------------------------------------
    # counts sözlüğü:
    # - Her path'in popülasyonda kaç kez tekrarlandığını tutar
    # - Fitness sharing için kullanılır
    #ordaki +1 elimdeki path
    

        fit_vals = [shared_fitness(p, counts, share_strength=0.15) for p in population]
    # -------------------------------------------------
    # Fitness hesaplama (fitness sharing dahil)
    # -------------------------------------------------
    # Her birey için shared_fitness hesaplanır
    # (QoS + ceza + klon paylaşımı)
    
        ranked = sorted(zip(population, fit_vals), key=lambda x: x[1])
        # Fitness'a göre sıralama (küçük daha iyi)
        

        new_population = [p for p, _ in ranked[:elite_size]]
    # -------------------------------------------------
    # Elit bireylerin korunması
    # -------------------------------------------------
    # En iyi elite_size adet birey
    # doğrudan yeni nesle aktarılır
    
        target_fill = pop_size - inject_count
        # Enjeksiyon öncesi hedef popülasyon boyutu
        

        while len(new_population) < target_fill:
    # -------------------------------------------------
    # YENİ BİREY ÜRETİM DÖNGÜSÜ (Breeding Loop)
    # -------------------------------------------------
    # Amaç:
    # - new_population listesini hedef sayıya (target_fill) kadar doldurmak
    # - Her adımda "yeni bir çocuk birey (child path)" üretmek
    
    # Neden while?
    # - Üretilen çocuk bazen geçersiz olabilir (kopuk path, demand ihlali, döngü vb.)
    # - Geçersiz bireyleri atlayıp (continue) tekrar üretim yapmak gerekir
    # - Bu yüzden "kaç deneme yapılacağını" değil "kaç geçerli birey toplandığını"
    #   kontrol eden bir döngü gerekir
    
    # target_fill = pop_size - inject_count olduğu için:
    # - Popülasyonun büyük kısmı GA operatörleriyle üretilir
    # - Kalan kısım daha sonra random immigrant ile doldurulur (diversity injection)
    
            p1 = rank_roulette_selection(population, fit_vals)
            p2 = rank_roulette_selection(population, fit_vals)
   # -------------------------------------------------
   # SEÇİLİM (Selection) – Ebeveyn seçimi
   # -------------------------------------------------
   # Burada iki ebeveyn seçiyoruz: p1 ve p2
   
   # rank_roulette_selection ne yapıyordu?
   # - Fitness değeri "sıralama"ya çevrilir (rank-based)
   # - En iyi bireyler daha yüksek seçilme olasılığı alır
   # - Ama orta/kötü bireylerin olasılığı sıfırlanmaz
   
   # Neden iki ebeveyn?
   # - Crossover operatörü iki bireyin genlerini karıştırmak için
   #   iki kaynağa ihtiyaç duyar
   
   # Neden popülasyon + fit_vals veriyoruz?
   # - pop = birey havuzu
   # - fit_vals = her bireyin fitness değeri (shared fitness dahil)
   # - Seçim bu değerlere göre olasılıklı yapılır
            

            if rnd.random() < crossover_rate:
                child = crossover_path_aware(p1, p2)
            else:                
                child = p1[:]
    # -------------------------------------------------
    # ÇAPRAZLAMA (Crossover) – Çocuk üretimi
    # -------------------------------------------------
    # rnd.random() 0.0–1.0 arası bir sayı döndürür
    # Eğer bu sayı crossover_rate'ten küçükse:
    # - crossover uygulanır
    
    # crossover_rate = 0.9 ne demek?
    # - Çocukların %90'ı iki ebeveynin karışımı olacak
    # - %10'unda ise crossover yapılmayacak, sadece kopya alınacak
    
    # Neden bazen crossover yapmıyoruz?
    # - GA’da sadece crossover'a bağımlı kalmak risklidir
    # - Bazı iyi bireylerin yapısını koruyarak nesle taşımak gerekir
    # - Ayrıca crossover bazen geçersiz path üretir (kopuk zincir)
    #   → bu risk çok yükselmesin diye küçük bir "kopyalama" payı bırakılır
    
    # crossover_path_aware(p1,p2) ne sağlar?
    # - Ortak düğüm üzerinden birleşim yapmaya çalışır
    # - Böylece child path’in kopuk olma ihtimali daha az olur
    
    # else: child = p1[:]
    # - Crossover yoksa p1'in bir kopyası alınır
    # - p1[:] kullanımı:
    #   * Referans kopya değil, gerçek liste kopyası üretir
    #   * Sonradan child üzerinde değişiklik yapılınca p1 bozulmaz

            if rnd.random() < mutation_rate:
                child = mutate(child)
    # -------------------------------------------------
    # MUTASYON (Mutation) – Rastgele değişim
    # -------------------------------------------------
    # mutation_rate = 0.30 ne demek?
    # - Üretilen çocukların %30'una mutasyon uygulanır
    
    # Neden mutasyon şart?
    # - Crossover sadece mevcut genleri yeniden karıştırır
    # - Mutasyon yeni genetik bilgi üretir:
    #   * yeni düğümler
    #   * yeni yol segmentleri
    # - Klonlaşmayı azaltır
    # - Yerel minimumdan çıkmayı sağlar
    
    # mutate(child) içinde ne oluyordu?
    # - %50 ihtimalle tek gen (node) değiştiriliyor (küçük değişim)
    # - %50 ihtimalle segment reset yapılıyor (büyük değişim)
    # Bu sayede:
    # - Hem küçük iyileştirmeler (exploitation)
    # - Hem yeni bölgeleri keşif (exploration)
    # aynı anda mümkün olur
        

            child = repair_path(child) if child else None
    # -------------------------------------------------
    # ONARIM (Repair) – Geçersiz path düzeltme
    # -------------------------------------------------
    # Crossover ve mutation sonrası child:
    # - Kopuk olabilir (edge yok)
    # - Demand ihlali yapabilir
    # - Döngü içerebilir
    
    # Bu projede "çocuğu direkt çöpe atmak" yerine
    # repair mekanizmasıyla kurtarmaya çalışıyoruz:
    
    # repair_path(child) mantığı:
    # - child’in başından itibaren geçerli olan prefix kısmı korur
    # - bozulduğu noktada durur
    # - oradan hedefe (D) random_walk_to_target ile bağlamayı dener
    
    # Neden repair iyi bir şey?
    # - Ebeveynlerden gelen yararlı kısım kaybolmaz
    # - GA’nın ürettiği birey sayısı düşmez
    # - Random walk ile “deterministik shortest-path” kullanmadan onarır
    
    # if child else None:
    # - child zaten None ise repair çağırma
    # - Güvenlik amaçlı
        
            if not child:
                continue
    # -------------------------------------------------
    # GEÇERSİZ / ONARILAMAYAN ÇOCUKLARI ATLA
    # -------------------------------------------------
    # Eğer repair başarısız olursa repair_path None döndürür
    # Bu durumda:
    # - Bu çocuk popülasyona EKLENMEZ
    # - Döngü başa döner ve yeni bir child üretmeye çalışılır
    
    # Neden continue kullanıyoruz?
    # - "Hard delete" popülasyondan birey silmek demek değil
    # - Burada sadece BAŞARISIZ üretimi kabul etmiyoruz
    # - GA’nın üretim hattında kalite kontrol gibi düşün:
    #   * geçersiz ürün raflara girmiyor
    #   * üretim devam ediyor
        

            if tuple(child) in set(tuple(x) for x in new_population):
    # -------------------------------------------------
    # KLON KONTROLÜ (Duplicate / Clone Detection)
    # -------------------------------------------------
    # Burada şunu kontrol ediyoruz:
    # - Üretilen child path,
    # - yeni popülasyonda HALİHAZIRDA var mı?
    
    # Neden tuple(child)?
    # - Python listeleri set/dict anahtarı olamaz (mutable)
    # - Path'i tuple'a çevirerek hashlenebilir hale getiriyoruz
    
    # set(tuple(x) for x in new_population):
    # - Mevcut popülasyondaki tüm path'leri
    #   hızlı kontrol edilebilir bir küme haline getirir
    
    # Bu kontrol:
    # - "Bu child birebir aynı mı?" sorusunu sorar
    # - Benzerlik değil, TAM AYNI olmayı kontrol eder
    
                if rnd.random() < 0.15:
                    new_population.append(child)
        # ---------------------------------------------
        # KLONU KABUL ET (DÜŞÜK OLASILIKLA)
        # ---------------------------------------------
        # %15 ihtimalle:
        # - Aynı child kabul edilir
        
        # Neden tamamen yasaklamıyoruz?
        # - GA'da klonlar her zaman KÖTÜ değildir
        # - Çok iyi bir çözüm birkaç kez hayatta kalabilir
        
        # Ama:
        # - Bu oran BİLİNÇLİ olarak düşük tutulur
        # - Klonların popülasyonu domine etmesi zorlaştırılır
        
                else:
                    continue
        # ---------------------------------------------
        # KLONU REDDET
        # ---------------------------------------------
        # %85 ihtimalle:
        # - Bu child popülasyona alınmaz
        # - Yeni bir child üretmek için döngü başa döner
        
        # Bu yaklaşım:
        # - Hard delete (zorla silme) değildir
        # - Ama üretim sürecinde klonları baskılar
        
            else:
                new_population.append(child)
    # -------------------------------------------------
    # YENİ VE BENZERSİZ BİREY
    # -------------------------------------------------
    # Eğer child popülasyonda YOKSA:
    # - Hiç tereddüt edilmeden eklenir
    
    # Bu:
    # - Çeşitliliği doğrudan artırır
    # - Yeni arama bölgelerinin keşfini sağlar
    
# -------------------------------------------------
# KLON BASKILAMA STRATEJİSİNİN ÖZETİ
# -------------------------------------------------
# - Klonları tamamen yasaklamıyoruz (hard constraint yok)
# - Ama çoğalmalarını istatistiksel olarak zorlaştırıyoruz

# Kullanılan yöntemler:
# Fitness sharing (daha önce)
# Üretim sırasında düşük kabul olasılığı (%15)

# Sonuç:
# - Popülasyon tek bir path'e çökmez
# - GA'nın stokastik doğası korunur
# - Akademik olarak savunulabilir bir yaklaşım


    
    

        tries_inject = 0
        while len(new_population) < pop_size and tries_inject < 300:
            tries_inject += 1
            rp = random_simple_path()
# -------------------------------------------------
# ÇEŞİTLİLİK ENJEKSİYONU – RANDOM IMMIGRANTS
# -------------------------------------------------
# Bu döngünün amacı:
# - GA operatörleriyle doldurulamayan boşlukları
#   rastgele yeni bireylerle doldurmak

# tries_inject:
# - Sonsuz döngüyü engellemek için güvenlik sayacı
# - En fazla 300 deneme yapılır

# random_simple_path():
# - S → D arasında
# - Shortest-path KULLANMADAN
# - Random walk ile yeni bir yol üretir

            if rp and is_valid_path(rp):
                if tuple(rp) not in set(tuple(x) for x in new_population) or rnd.random() < 0.20:
                    new_population.append(rp)
# -------------------------------------------------
# ENJEKSİYON BİREYİNİN KABULÜ
# -------------------------------------------------
# rp geçerliyse:
# - Önce yine klon kontrolü yapılır

# Eğer rp popülasyonda YOKSA:
# - Direkt kabul edilir
#
# Eğer rp popülasyonda VARSA:
# - %20 ihtimalle yine de kabul edilir

# Neden %20?
# - Enjeksiyonun amacı çeşitlilik
# - Ama tamamen deterministik "sadece yeni" kuralı
#   popülasyonu doldurmayı zorlaştırabilir

# Bu esneklik:
# - Popülasyonun dolmasını garanti eder
# - Ama klon baskısını yine korur
    

        while len(new_population) < pop_size:
            new_population.append(rnd.choice(population)[:])
# -------------------------------------------------
# GÜVENLİK DOLGUSU (FINAL FALLBACK)
# -------------------------------------------------
# Çok nadiren:
# - Ne GA operatörleri
# - Ne random immigrants
# popülasyonu tam dolduramayabilir

# Bu durumda:
# - Eski popülasyondan rastgele kopyalar alınır

# Neden kopya?
# - GA çökmemeli
# - Bir sonraki nesil mutlaka pop_size birey içermeli

# rnd.choice(population)[:]:
# - Rastgele birey seçilir
# - [:] ile gerçek kopya alınır (referans değil)

# Bu adım:
# - Algoritmanın sağlamlığını artırır
# - Normalde çok nadir tetiklenir
    

        population = new_population
# -------------------------------------------------
# NESİL DEĞİŞİMİ
# -------------------------------------------------
# Yeni oluşturulan popülasyon,
# bir sonraki neslin popülasyonu olur

# GA'nın evrim döngüsü burada tamamlanır
        
    if __name__ == "__main__":
        print(population[::-1])
# -------------------------------------------------
# Debug / test amaçlı çıktı
# -------------------------------------------------
# Modül direkt çalıştırıldığında
# popülasyonun ters sıralı hali yazdırılır


    best = min(population, key=lambda p: base_fitness(p))
# -------------------------------------------------
# EN İYİ BİREYİN SEÇİLMESİ
# -------------------------------------------------
# Son popülasyon içindeki
# fitness değeri EN KÜÇÜK olan birey seçilir

# Bu problem bir MINIMIZATION problemidir:
# - Fitness ↓  → çözüm ↑

    return best, path_metrics(G, best)
# En iyi path ve QoS metrikleri döndürülür


if __name__ == "__main__":
    node_csv = "BSM307_317_Guz2025_TermProject_NodeData.csv"
    edge_csv = "BSM307_317_Guz2025_TermProject_EdgeData.csv"
    demand_csv = "BSM307_317_Guz2025_TermProject_DemandData.csv"

    G, demands = load_graph_and_demands(node_csv, edge_csv, demand_csv)

    best_path, metrics = ga_find_path(G, 0, 249)

    total_val = total_cost(metrics[0], metrics[1], metrics[2], (1/3, 1/3, 1/3))

    print("Best path:", best_path)
    print("Metrics (delay, reliability cost, resource cost):", metrics)
    print("Total Cost:", total_val)
