# boyut-indirgeme-dimensionality-reduction
*Sınıflandırma problemleri için meta-sezgisel boyut azaltma aracının tasarımı ve uygulaması (Bitirme Tezi)*

# İçindekiler
1. [Özet](#özet)
2. [Genel Bilgiler](#genel-bilgiler)
   - [Giriş](#giriş)
   - [Tezin Amacı](#tezin-amacı)
   - [Tezin Organizasyonu](#tezin-organizasyonu)
3. [Yöntem](#yöntem)
   - [Meta-Sezgisel Arama Süreci](#meta-sezgisel-arama-süreci)

# Özet

Sınıflandırma problemleri yapay zekanın geniş bir uygulama alanını temsil
etmektedirler. Sınıflandırma problemlerinin modellenmesi ve çözümlenmesi için istatistiksel,
örnek tabanlı, ağaç yapısını esas alan, yapay sinir ağları temelli farklı özelliklerde ve farklı
yeteneklerde birçok algoritma geliştirilmiştir. Sınıflandırma problemlerinin özelliklerine bağlı
olarak algoritmaların performansları değişebilmektedir. Bir sınıflandırma probleminde veri
örneği sayısının değişmesi yalın-bayes sınıflandırıcının cevap üretme süresi üzerinde hiçbir
etkiye sahip değilken k-nn sınıflandırıcının cevaplama hızını ise doğrudan etkilemektedir.
Dolayısıyla sınıflandırma problemlerine ilişkin bazı özellikler algoritmaların performansları
üzerinde önemli etkiye sahiptirler. Bunun yanında bir sınıflandırma probleminin boyutunun ya
da nitelik sayısının değişmesi durumunda performansı bundan etkilenmeyen bir algoritma
yoktur. Hemen tüm algoritmalar için problem boyutunun artması performansı olumsuz yönde
etkileyen bir faktördür. Dolayısıyla sınıflandırma problemlerinde problem boyutu azaltma
üzerinde yoğun bir şekilde çalışılan önemli bir konudur. Özellikle yüksek boyutlu problem
uzaylarında problem için en uygun modeli yaratmak zor bir görevdir. Karmaşıklık düzeyi
yüksek arama uzaylarında genel en iyi çözümü bulmak olanaksız olarak tanımlanmaktadır.
Dolayısıyla bu tür arama problemleri için en ideal çözüm yolu meta-sezgisel arama
algoritmalarından faydalanmaktır. Bu tez çalışmasında sınıflandırma problemlerinde boyut
azaltmak için AGDE (adaptive guided differential evolution) algoritması temelli meta-sezgisel
nitelik seçim yöntemi geliştirilmektedir. Geliştirilen yöntem niteliklerin MSA’larla
ağırlıklandırılmasını ve eşik değerden küçük ağırlıklı niteliklerin tespit edilmesini
sağlamaktadır. Geliştirilen algoritma, UCI Machine Learning veri havuzunda yer alan dört
farklı veri seti üzerinde tatbik edilmiştir. Deneysel çalışmalardan elde edilen sonuçlar AGDE-
tabanlı nitelik seçim yönteminin sınıflandırma problemleri için etkili bir boyut azaltma aracı
olduğunu göstermektedir.

# Genel Bilgiler
  ## Giriş
  Nitelik sayısı fazla ve karmaşıklık düzeyi yüksek yapay zekâ problemleri için etkili
çözümler geliştirmek zordur. Bu tür zor problemler için bir yapay zekâ çözümü geliştirmeden
önce problem modelinin en etkili şekilde oluşturulmasına ihtiyaç vardır. Bunun için problemi
tanımlayan niteliklerin (bağımsız değişkenlerin) hedef parametre (bağımlı değişken) üzerindeki
etkilerinin belirlenmesine (ağırlıklandırılmasına) ihtiyaç vardır. Hedef parametre üzerinde
yeterince etkiye sahip olmayan nitelikler belirlendikten sonra bunların problem modelinden
çıkartılması mümkün olabilir. Bu süreç problemin boyutunun azaltılması ve problemi
modellemek için en uygun niteliklerin belirlenmesi olarak ifade edilebilir. Bu süreçte problem
niteliklerinin ağırlıklandırılması için kestirim yapılmasına yani optimizasyon çalışmasına
ihtiyaç vardır. Optimizasyon çalışmaları neticesinde, problem için kabul edilebilir bir
sınıflandırma performansı sağlayan ve asgari sayıda nitelikten oluşan bir modelin yaratılması
sağlanabilir. Optimizasyon sürecinde ise problem boyutunun ve karmaşıklık düzeyinin fazla
olması, meta-sezgisel arama algoritmalarının kullanılması gerekliliğini ortaya koymaktadır.
Meta-sezgisel arama (MSA) algoritmaları, yüksek karmaşıklığa sahip
optimizasyon/kestirim çalışmalarında en etkili yöntemlerdirler. MSA algoritmaları kullanılarak
farklı alanlardan binlerce optimizasyon çalışması yürütülmüştür. MSA’lar sadece optimizasyon
sürecinde arama algoritmaları olarak değil birçok çalışmada daha güçlü ve etkili melez yapay
zekâ algoritmaları tasarlamak için de kullanılmaktadırlar. Bu amaçla tahmin [1], sınıflandırma
[2-4] ve kümeleme [5-9] problemlerini çözümlenmesi için geliştirilmiş MSA-tabanlı melez
algoritmalar mevcuttur. MSA’lar doğadan esinlenilerek geliştirilmiş tekniklerdir. 1950’li
yıllardan bu yana çok sayıda MSA algoritması geliştirilmiştir [10-14]. Son on yılda ise bu sayı
her yıl onlarca yeni MSA algoritmasının geliştirilmesi ile artmaktadır. Geliştirilen yüzlerce
algoritma arasından en etkili olanı belirlemek ise zor bir iştir. Çünkü tüm problemlerde en iyi
olan bir MSA tekniği yoktur.
MSA’ların performansları problem bağımlıdır. Bu doğada da böyledir. Doğada her canlı ya da
süreç kendi işleyişinde kusursuz bir performans sergilerken bir başka görev ya da işleyiş için
çok başarısız olabilmektedir. Bunun yanında son dönemde geliştirilen modern MSA
algoritmaları pek çok tipte farklı özellikteki optimizasyon problemleri üzerinde test edilerek
geliştirildikleri için nispeten daha kararlı ve başarılı bir arama performansına sahip
olmaktadırlar.
Bu tez çalışmasında optimizasyon sürecinde kullanılan arama algoritmalarından biri
oldukça yeni ve etkili bir MSA tekniği olan Adaptive Guided Differantial Evaluation (AGDE)
dir [15]. AGDE, adından da anlaşılacağı üzere Differential Evaluation (DE)’nin [16]
performansı iyileştirilmiş bir varyasyonu olarak geliştirilmiş algoritmadır. DE’nin sahip olduğu
güçlü temeller AGDE’nin de başarısındaki en büyük etkenlerdir.
Tez çalışmasında birçok farklı meta-sezgisel yöntem kullanılarak gerçekleştirilmiş olan
[17-29] ve literatürde boyut indirgeme (dimensionality reduction) olarak bilinen konu üzerine
çalışılmaktadır. Bu amaçla melez bir yapay zekâ algoritması geliştirilmiş ve problem uzayı
bağımsız değişkenleri sayısal veri tipinde olan sınıflandırma problemleri olarak belirlenmiştir.
Amaç bu tipteki problemlere ait veri setlerini kullanarak sınıflandırma performansını korurken
nitelik sayısını azaltmaktır (boyut indirgeme). Sınıflandırma algoritması olarak k-en yakın komşu (k-nearest neighbor, k-nn) ve sezgisel k-nn algoritmaları kullanılmıştır. MSA
algoritması olarak ise güncel ve güçlü bir MSA algoritması olan AGDE kullanılmıştır. Bu
amaçla geliştirilen boyut indirgeme algoritması bir optimizasyon yöntemi olmanın da ötesinde
birden fazla algoritmanın bir arada kullanıldığı melez bir yöntemdir. Bu melez algoritma, k-nn
sınıflandırıcı için en iyi k-değerinin belirlenmesi, sınıflandırma eşik değerinin tanımlanması,
probleme ait niteliklerin ağırlıklandırılması ve sezgisel sınıflandırma gibi farklı gereksinimlere
cevap veren yeteneklere sahip olarak geliştirilmiştir. Tez çalışması için ihtiyaç duyulan
sınıflandırma problemlerine ait veri setleri için de UCI Machine Learning veri havuzu
kullanılmıştır [30-33].
  ## Tezin Amacı
  Tezin amacı, çok boyutlu sınıflandırma problemleri için uygulama alanına bağlı
kalmaksızın boyut indirgeme işlemini etkili bir şekilde yerine getiren meta-sezgisel
optimizasyon tabanlı bir boyut indirgeme aracı geliştirmektir. Sınıflandırma problemleri için
meta-sezgisel tabanlı boyut indirgeme algoritması geliştirilirken boyut azaltıldıktan sonraki
sınıflandırma başarısının düşmemesi amaçlanmıştır.
  ## Tezin Organizasyonu
  Tez çalışması bölümleri sırasıyla, yöntem, deneysel çalışma ve sonuçlar şeklinde
tasarlanmıştır. Yöntem bölümünde öncelikle, bu çalışmada geliştirilen melez boyut indirgeme
algoritmasının temel öğeleri olan k-nn sınıflandırıcı ve AGDE algoritması tanıtılmaktadır. Daha
sonra önerilen yöntem adım-adım açıklanmıştır. Deneysel çalışma bölümünde ise dört farklı
probleme ait veri setleri kullanılarak geliştirilen melez algoritmanın bu problemler üzerindeki
performansı test edilmiştir. Son olarak bu çalışmadan elde edilen sonuçlar değerlendirilmiştir.
# Yöntem
  Yöntem bölümünde öncelikle veri madenciliği konusu kısaca ele alınmakta ve sırasıyla
meta-sezgisel arama süreci, k-nn algoritması, AGDE algoritması ve önerilen yöntem hakkında
bilgi verilmektedir.
  ## Meta-Sezgisel Arama Süreci
  Yapay zekânın bir türü olan Meta-sezgisel Arama (MSA) algoritmalarını geliştirme
çalışmaları 1950’li yıllara dayanmaktadır. Michigan üniversitesinde Prof. John Holland ve
öğrencilerinin geliştirdikleri Genetik Algoritma bu çalışmalara hız kazandırmıştır [42]. Son
yıllarda gerek algoritma geliştirme çalışmalarında gerekse de problemlere tatbik edilmeleri
hususunda elde edilen başarılar meta-sezgisel algoritmaların önemini giderek artırmaktadır [43-
47]. MSA algoritmaları yapay zekanın uygulandığı her alanda, tahmin, kümeleme,
sınıflandırma gibi problemleri çözümlemek için melez algoritmaların geliştirilmesinde ve asıl
olarak optimizasyon problemlerinin çözümünde yaygın bir şekilde kullanılmaktadırlar.
Maliyetleri azaltmanın ve verimliliği artırmanın kritik önem kazandığı çağımızda süreçleri ve
sistemleri optimum şekilde modellemenin etkili yollarından biri olarak meta-sezgisel
algoritmalara başvurulmaktadır.
