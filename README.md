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
   - [k-nn Sınıflandırma Algoritması](#k-nn-sınıflandırma-algoritması)
   - [AGDE Sezgisel Arama Algoritması](#agde-sezgisel-arama-süreci)
   - [Önerilen Yöntem: Sezgisel Boyut İndirgeme](#önerilen-yöntem)
4. [Kaynaklar](#kaynaklar)

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
algoritmalara başvurulmaktadır.<br>
   Enerji, inşaat, pazarlama, üretim, bilgi teknolojileri, havacılık ve uzay sanayii gibi birçok
alanda binlerce sistem ve sürecin optimizasyonunda ve Endüstri 4.0 gibi modern otomasyon
sistemlerinin ve uygulamalarının geliştirilmesinde meta-sezgisel optimizasyon tekniklerinden
faydalanılmaktadır. Büyük ve karmaşık bir problem uzayında arama yapmanın etkili bir yolu
MSA algoritmalarını kullanmaktır. MSA algoritmalarının doğadan kaynaklı bileşenlerinin
yetenekleri ve özellikleri farklı olsa da bir meta-sezgisel arama süreci temel olarak aynı
adımlardan oluşur. Bu adımlar Algoritma 1’de verilmektedir.<br><br>
**Algoritma 1.** *Meta-sezgisel arama sürecinin temel adımları*
```
i) Problemin yaratılması (uygunluk fonksiyonunun, ceza fonksiyonunun tanımlanması)
ii) Çözüm adayının tasarımı ve çözüm adayları topluluğunun yaratılması
iii) Adayların uygunluk değerlerinin hesaplanması
iv) İteratif süreç (buluşsal arama)
   - Komşuluk Araması
   - Çeşitliliğin Sağlanması
   - Çözüm Adayı Setinin Güncellenmesi
v) Sonlandırma kriteri sağlandı mı?
   - Hayır (Adım iv'e dön)
   - Evet (arama sürecini sonlandır ve en iyi çözüm adayını kaydet)
```
   Algoritma 1’de verilen (i, ii, iii ve v) numaralı adımlar bütün MSA algoritmaları için
aynıdır. (iv) numaralı adım ise tüm MSA için farklıdır. (iv) numaralı adımda arama
algoritmasına özgü operatörler/işlemler uygulanmaktadır. Arama sürecinin başarısı bu
operatörlerin yeteneklerine bağlıdır. (iv) numaralı adımda verilen çözüm adayları topluluğunun
güncellenmesinde temel olarak iki farklı yöntem kullanılmaktadır. Bunlar çözüm adaylarının
arama uzayına normal/gauss dağılımı ile yerleştirilmesi ve rastgele yerleştirilmesidir. MSA
algoritmaları ile ilgili çalışmalar incelendiğinde bu dağılım tiplerinden biriyle algoritmaların
geliştirildiği ve test edildiği görülmektedir.
   ## k-nn Sınıflandırma Algoritması
   k-nn algoritması uzaklığa dayalı sınıflandırma algoritmasıdır. Sınıflandırma, önceden elde
ettiğimiz bilgilerin veya verilerin hangi sınıftan olduğu biliniyorsa, yeni gelen verinin hangi
sınıfa ait olacağının belirlenmesi işlemidir. Uzaklığa dayalı algoritmalardan en bilineni ve en
yaygın kullanılanıdır. Sınıflandırma yapılırken eldeki verilerin birbirlerine olan uzaklığı veya
benzerliği kullanılır. Veriler arasındaki mesafe ölçülülerken en çok kullanılan yöntemler Öklit,
Manhattan ve Minkovski uzaklık metrikleridir. k-nn algoritmasında öncelikle gözlemler
arasındaki uzaklıklar belirlenir. Gözlemler arasındaki uzaklıkların belirlenmesinde herhangi bir
uzaklık bağıntısı kullanılabilir. Sınıflandırma yapılırken veri tabanındaki her bir kaydın diğer
kayıtlara olan uzaklığı hesaplanır.<br><br>
**Algoritma 2.** *k-nn algoritmasının sözde kodu [3]*
```
1. Başla
2. Veri setinin tanımlanması: probleme ait n-adet örnek gözlemleri içeren ve problem
uzayını temsil etme kabiliyeti yüksek (gözlem uzayını homojen olarak
örnekleyen) X veri setini oluştur.
3. Uzaklık bağıntısının belirlenmesi: gözlemler arasındaki uzaklıkların
hesaplanmasında kullanılacak yöntemi belirle.
4. k-değerinin belirlenmesi: gözlem sayısına ve veri setinin karakteristiğine bağlı
olarak k-komşu sayısı için arama uzayının sınırlarını tanımla.
5. for each kj
      kj için sınıflandırma performansını SPkj = fk-nn(kj)
      if (SPkj > SPkj-1)
         k= kj
      end if
   end
6. En iyi sınıflandırma performansı sağlayan k-değerini kaydet
7. Sınıf etiketi belirlenecek olan q sorgu gözlemini tanımla
8. for i=1:n
      D[i]=q ile Xi arasındaki uzaklığı hesapla
   end
9. Xq[k]=D[i] uzaklık dizisinden q sorgu gözlemine en yakın k-adet gözlemi belirle
10. Xq[k] gözlemlerinin sınıflarını dikkate alarak çoğunluk oylaması/ağırlıklı
oylama yöntemi kullanarak q-gözleminin sınıfını belirle.
11. Bitir
```
Klasik k-nn algoritmasında gözlemler arasındaki uzaklık hesabı öklit bağıntısı
kullanılarak Eşitlik 1’de verildiği gibi hesaplanır [3].

![oklid_formula](/images/oklid_formul.png)

Eşitlik 1’de sınıflandırma probleminin nitelik sayısı n-ile, X1 ve X2 gözlemleri ise <x11, x12, x13,
...,x1n> ve <x21, x22, x23, ...,x2n> ile temsil edilmektedir. Buna göre uzaklık hesabında
problemin niteliklerinin eşit derecede etkiye sahip olduğu görülmektedir. Bunun yanında her
bir niteliğin alt ve üst sınır değerlerine bağlı olarak sayısal büyüklüklerinin de uzaklık hesabı
üzerinde farklı bir etki yaratabileceği dikkate alınmalıdır. Bu durum niteliklerin birbirlerine
baskınlık kurması olarak nitelendirilir. Baskınlık durumunu bertaraf etmek içi her bir niteliğin
normalizasyonu [0, 1] arasında ölçeklendirilmesi mümkündür. Ancak bu durumda ise probleme
ait niteliklerin, gözlemlerin sınıflandırılmasında eşit derecede etkiye sahip olmaları gibi bir
durum ortaya çıkar. Örneğin bir müşteriye verilecek banka kredisi hesaplanırken kişinin
mesleği, aylık geliri, kredi skoru, borcu, cinsiyeti, yaşı, yaşadığı şehir, varlıkları gibi bilgiler
dikkate alınabilir. Ancak bu niteliklerin her biri kişiye verilecek kredinin miktarının
belirlenmesinde farklı derecede öneme sahiptir. Dolayısıyla klasik k-nn algoritmasının işleyişi
incelendiğinde, sınıflandırma sürecinde niteliklerin etkilerinin eşit olduğunun kabul edilmesi
doğru ve etkili bir yaklaşım değildir. Bu nedenle niteliklerin önem/etki derecelerinin
belirlenmesi için sezgisel arama algoritmalarının kullanıldığı yöntemler geliştirilmiştir. En iyi
sınıflandırma performansının elde edildiği ağırlık değerlerinin arandığı bu yöntemler literatürde
oldukça yaygın bir şekilde kullanılmaktadırlar. Takip eden bölümde, sezgisel sınıflandırma
algoritmasının iki temel öğesinden biri olan meta-sezgisel arama algoritmalarının güncel bir
örneği tanıtılmaktadır.
   ## AGDE Sezgisel Arama Süreci
   AGDE algoritması, popülasyon tabanlı sezgisel bir optimizasyon tekniği olmakla birlikte
global çözümü bulma açısından basit ama aynı zamanda da güçlü bir tekniktir [15]. AGDE
algoritmasının temeli DE (differential evolution) algoritmasına dayanmaktadır. DE algoritmasının [16] temeli ise genetik algoritmaya [10] dayanır. Algoritmanın adımları
Algoritma 3’te verilmiştir.<br><br>
**Algoritma 3.** *AGDE algoritmasının temel ddımları [15]*
```
i) Problemin yaratılması (uygunluk fonksiyonunun, ceza fonksiyonunun tanımlanması)
ii) Çözüm adayının tasarımı ve çözüm adayları topluluğunun yaratılması
iii) Çözüm adayların uygunluk değerlerinin hesaplanması
iv) İteratif süreç (sonlandırma kritesi sağlanıncaya kadar devam et: amaç fonksiyonu azami
değerlendirme sayısı)
      - Seçim süreci: Popülasyondan r1 ≠ r2 ≠ r3 ≠ i olmak üzere üç çözüm
adayını rastgele seç
      - Mutasyon
      - Çaprazlama
v) Sonlandırma kriteri sağlandı mı?
      - Hayır (Adım iv’e dön)
      - Evet (Arama sürecini sonlandır ve en iyi çözüm adayını kaydet)
```
Algoritma 3’te verilen bilgilere göre DE algoritmasında arama süreci yaşam döngüsü
seçim işlemi, çaprazlama ve mutasyon olmak üzere üç adımdan oluşur. AGDE algoritmasında
da bu üç adım ortaktır fakat AGDE algoritmasının geliştirilmesinde çeşitlilik yeteneğinin
iyileştirilmesine yönelik olarak yeni bir mutasyon tekniği önerilmiştir. Bu amaçla Algoritma
3’te seçim sürecinde tanımlanan r1, r2 ve r3 çözüm adaylarının seçilme yöntemleri DE
algoritmasından farklı şekilde uygulanmaktadır. Bu üç çözüm adayının seçilme süreci ve
AGDE algoritmasında işlevleri hakkında detaylı bilgi almak için referans çalışma incelenebilir
[15]. AGDE algoritmasında çalışılan ikinci konu ise arama sürecinde komşuluk araması ve
çeşitlilik arasındaki dengenin sağlanması için yeni bir adaptasyon şemasının önerilmesidir.
AGDE’nin arama performansı mutasyon ve çaprazlama operatörlerinin başarısına bağlıdır.
Çaprazlama sürecinin etkisi ise çaprazlama oranına (CR) bağlıdır. CR’nin büyük olması arama
sürecinde çeşitliliğe katkı sağlarken hassas aramayı ise engellemektedir. Bu durum genetik
algoritma ve diferansiyel evrim algoritmasında da aynıdır. CR’nin küçük olması halinde ise
komşuluk araması hassas bir şekilde yerine getirilmekte ancak bu defa da yerel çözüm
tuzaklarına yakınsama problemi ortaya çıkabilmektedir. Bu nedenlerden ötürü, AGDE
algoritmasında CR parametresinin problem tipine ve arama sürecinin gereksinimlerine bağlı
olarak dinamik bir şekilde ayarlanması ihtiyacına yönelik çözüm geliştirilmiştir. Eşitlik 2’de
çaprazlama oranının dinamik bir hale getirilmesi için jenerasyon sayısına bağlı olarak
geliştirilmiş ve rastgeleliği içeren bir yöntem verilmektedir. AGDE algoritmasında çaprazlama
oranı CR değerlerini üretmek için, önceden belirlenmiş bir aday havuzu kullanır. Bu havuzlar
CR1 ∈ [0.05, 0.15]; CR2 ∈ [0.9, 0.1].

![cr_formula](/images/cr_formula.png)

Eşitlik 2’de verilen rand(0,1) ifadesi uniform dağılıma sahip random bir değer üretmek
için ve p1 ve p2 ise çaprazlama oranı havuzundaki herhangi bir CR setinin seçilme olasılıklarını
göstermektedir. Sonuç olarak AGDE algoritması geleneksel DE yönteminden farklı olarak
çaprazlama oranının dinamik bir şekilde ayarlandığı yeni bir yöntemi uygulayarak daha etkili
bir arama performansı sergilemektedir. Daha fazla bilgi edinmek için lütfen referans çalışmayı
inceleyiniz [15].
   ## Önerilen Yöntem
   Sezgisel boyut indirgeme için geliştirilen AGDE-knn algoritmasının işleyişi Şekil 1’de
verilmektedir. Buna göre sınıflandırma problemlerinde sezgisel boyut indirgeme süreci birkaç
adımdan oluşmaktadır. Bu adımlar sırasıyla, probleme ait örnek bir veri setinin tanımlanması,
veri seti için en uygun k-değerinin belirlenmesi, sezgisel sınıflandırma algoritması kullanılarak
niteliklerin ağırlıklandırılması, boyut indirgeme için eşik değerlerin tanımlanması ve indirgeme
işlemi sonrası sınıflandırma performansının test edilmesi şeklinde ifade edilebilir. Aşağıda
sırasıyla Şekil 1’de verilen öğeler ve süreçler açıklanmaktadır.

![uml-diagram](/images/uml-diagram.png)

*Veri Tabanı Katmanı*
Çalışmada dört farklı boyutta veri seti kullanılmaktadır. İlk olarak veri setleri incelenmekte,
gerekli düzenlemeler yapılmaktadır. Yüklenen veri seti iki ayrı alt veri setine bölünmüştür:
Eğitim seti ve test seti. Modelin eğitim setiyle eğitilmesi sağlanmakta ve test setiyle başarısı
ölçülmektedir. Veri setlerinde toplam örnek sayısının yaklaşık olarak %70’i eğitim, %30’u test
için kullanılmıştır.<br>
*Uygulama Katmanı*
**Optimum k değerini belirleme:** İlk olarak her bir veri seti k-nn algoritmasıyla tatbik
edilmektedir. Klasik k-en yakın komşu algoritmasında niteliklerin problem üzerindeki etkileri
aynı kabul edilmektedir. Optimum k-değerini belirleme modülünün sözde kodu Algoritma 4’te
verilmiştir.<br><br>
**Algoritma 4.** Optimum k-değeri belirlemenin sözde kodu
```
1. Başla
2. k-değeri arama uzayı tanımlanır (k1,k2,.....).
3. k-değeri arama uzayının alt ve üst sınırı tanımlanır [m,n].
4. for i=m:n
      ki – değeri ile k-nn sınıflandırıcı çalıştırılır.
      H: Bulunan sınıflandırma hata değerleri kaydedilir.
   end for
5. sayac = 6 olarak tanımlanır.
6. while (sayac > 6)
      for i=m:n
         if (Hi < Hi+1)
            enKucukHata = Hi olarak seçilir.
            sayac bir azaltılır (sayac = sayac -1).
         else
            sayac bir artırılır (sayac = sayac +1).
         end if
      end for
   end while
7. enKucukHata değeri elde edilirken kullanılan k değeri seçilir.
8. Bitir.
```
k-nn algoritmasıyla niteliklerin henüz çıkarılmadığı, boyut azaltmanın henüz gerçekleşmediği
klasik modelin sınıflandırma hata değerleri elde edilmektedir. Her bir veri setinde ilk olarak k
değeri arama uzayı tanımlanır. Daha sonra k değeri arama uzayının alt ve üst sınırları tanımlanır.
Alt sınırdan başlanarak k-değeri kullanılarak elde edilen hata değeri ile k-değerleri artırılıp
bulunan hata değerleri birbiriyle karşılaştırılmaktadır. Bu döngü, bulunan en düşük hata
değerinden sonraki 6 iterasyonda hata değerinde azalma olmadığı sürece devam etmektedir.
Yani elde edilen hata değerinden sonraki 6 iterasyonda k-nn sınıflandırıcının performansı
iyileşmediğinde, o hata değeri elde edilirken kullanılan k-değeri optimum k değeri olarak
belirlenmektedir. Her bir veri seti için k-nn algoritmasının ideal k-değeri belirlendikten sonra
niteliklerin ağırlıklandırılması işlemine geçilmektedir.<br>
**AGDE-knn algoritması ile niteliklerin ağırlıklandırılması:** Sezgisel k-nn algoritması ile
niteliklerin probleme etkisi incelenip buna göre ağırlıklandırılması işlemi yapılmaktadır. Bunun
için meta-sezgisel arama algoritmalarından biri olan AGDE algoritması kullanılmaktadır.
AGDE algoritmasında çözüm adayları problem niteliklerinin ağırlıklarıdır. Meta-sezgisel
arama algoritmasının çözüm adayları 0 ile 1 arasında olacak şekilde kısıtlanmıştır. Yani ideal
ağırlıklar 0 ile 1 arasında aranacaktır. Amaç fonksiyon ise sezgisel k-nn’dir. Ağırlıkların yani
çözüm adayların uygunluk değerlerini ölçmek için hedef (amaç) fonksiyonundan dönen
sınıflandırma hata değerine bakılmaktadır.<br><br>
**Algoritma 5.** *AGDE-knn algoritması ile niteliklerin ağırlıklandırılmasının sözde kodu*
```
1. Başla
2. P: Problemin n nitelikli ağırlık dizisini temsil eden çözüm adaylarından rastgele bir
popülasyon oluşturulur
3. for i=1:n
      F: Her bir çözüm adayının ağırlık dizisini sezgisel k-nn’e göndererek uygunluk
değeri (hata değeri) hesaplanır
   end for
4. //Arama süreci yaşam döngüsünün başlangıcı
5. while (G1’den Gmax’a (maksimum uygunluk değerine) kadar git)
      for i=1:n
         Rastgele mutasyon faktörü üretilir
         Üç tane vektör seçilir: bir tane rastgele (Xr), bir tane en iyi çözüm adayı (Xp_enİyi), bir tane en kötü çözüm adayı vektörü (Xp_enKötü)
         D: Mutasyon noktası rastgele belirlenir
         for j = 1:D
            Çaprazlama yapılır
            yeniÇözüm: Yeni çözüm adayı elde edilir
         end for
         if (F(yeniÇözüm) <= (F(Xp_enİyi))
            Xp_enİyi = yeniÇözüm
         else
            Xi+1'e çaprazlama işlemi yapılır
         end if
      end for
   Sonlandırma kriterine kadar yeni jenerasyon oluştur
6. Bitir
```
Yani ağırlıklar k-nn algoritmasında kullanılıp k-nn’nin sınıflandırma hata değerini minimum
yapan ağırlıkların aranması işlemi gerçekleşmektedir. İdeal çözüm adayların aranması işlemi
sonlandırma kriteri tamamlanıncaya kadar devam eder.<br>
**Eşik değere göre nitelik seçimi / boyut azaltma:** AGDE algoritmasının nitelikler için en
uygun ağırlıkları arama işleminin sonlandırılmasından sonra problemin boyut azaltma/nitelik
çıkarımı aşamasına geçilmektedir. Bu aşamada eşik değer kullanılır. 0 ile 1 arasında bulunan
ağırlıklardan, eşik değerden düşük ağırlığa sahip nitelikler çıkarılmaktadır.<br>
**Boyutu azaltılmış modelin performans ölçümü:** Niteliklerin çıkarılması işleminden sonra
modelin sınıflandırma performansına bakılır. Klasik k-nn algoritması ve sezgisel k-nn
algoritmalarının sınıflandırma hata değerlerine bakılmaktadır. Eğer bu performans hata
değerleri, nitelik çıkarılmadan önceki modelin sınıflandırma hata değerlerinden düşükse yani
sınıflandırma performansı düşmemiş hatta iyileşmişse başarı sağlanmış olacaktır.
# Kaynaklar
1. Kahraman, H. T., Bayindir, R., &amp; Sagiroglu, S. (2012). A new approach to predict the
excitation current and parameter weightings of synchronous machines based on genetic
algorithm-based k-NN estimator. Energy Conversion and Management, 64, 129-138.
2. Kahraman, H. T. (2016). A novel and powerful hybrid classifier method: Development
and testing of heuristic k-nn algorithm with fuzzy distance metric. Data &amp; Knowledge
Engineering, 103, 44-59.
3. Kahraman, H. T., Sagiroglu, S., &amp; Colak, I. (2013). The development of intuitive
knowledge classifier and the modeling of domain dependent data. Knowledge-Based
Systems, 37, 283-295.
4. Yilmaz, C., Kahraman, H. T., &amp; Söyler, S. (2018). Passive mine detection and
classification method based on hybrid model. IEEE Access, 6, 47870-47888.
5. Cerrada, M., Aguilar, J., Altamiranda, J., &amp; Sánchez, R. V. (2019). A hybrid heuristic
algorithm for evolving models in simultaneous scenarios of classification and clustering.
Knowledge and Information Systems, 61(2), 755-798.
6. Li, K., Cao, X., Ge, X., Wang, F., Lu, X., Shi, M., ... &amp; Chang, S. (2020). Meta-Heuristic
Optimization Based Two-stage Residential Load Pattern Clustering Approach
Considering Intracluster Compactness and Inter-cluster Separation. IEEE Transactions on
Industry Applications.
7. Yadav, M., &amp; Prakash, V. P. (2020). A Comparison of the Effectiveness of Two Novel
Clustering-Based Heuristics for the p-Centre Problem. In Advances in Data and
Information Sciences (pp. 247-255). Springer, Singapore.
8. Zhou, Q., Benlic, U., Wu, Q., &amp; Hao, J. K. (2019). Heuristic search to the capacitated
clustering problem. European Journal of Operational Research, 273(2), 464-487.
9. Aljarah, I., Mafarja, M., Heidari, A. A., Faris, H., &amp; Mirjalili, S. (2020). Multi-verse
optimizer: theory, literature review, and application in data clustering. In Nature-Inspired
Optimizers (pp. 123-141). Springer, Cham.
10. Booker, L. B., Goldberg, D. E., &amp; Holland, J. H. (1989). Classifier systems and genetic
algorithms. Artificial intelligence, 40(1-3), 235-282.
11. Ramos-Figueroa, O., Quiroz-Castellanos, M., Mezura-Montes, E., &amp; Schütze, O. (2020).
Metaheuristics to solve grouping problems: A review and a case study. Swarm and
Evolutionary Computation, 100643.
12. Kulkarni, A. J., Singh, P. K., Satapathy, S. C., Kashan, A. H., &amp; Tai, K. (Eds.). (2019).
Socio-cultural Inspired Metaheuristics (Vol. 828). Springer.
13. Sibalija, T. V. (2019). Particle swarm optimisation in designing parameters of
manufacturing processes: A review (2008–2018). Applied Soft Computing, 84, 105743.
14. Fausto, F., Reyna-Orta, A., Cuevas, E., Andrade, Á. G., &amp; Perez-Cisneros, M. (2020).
From ants to whales: metaheuristics for all tastes. Artificial Intelligence Review, 53(1),
753-810.
15. Lin, K. C., Zhang, K. Y., Huang, Y. H., Hung, J. C., &amp; Yen, N. (2016). Feature selection
based on an improved cat swarm optimization algorithm for big data classification. The
Journal of Supercomputing, 72(8), 3210-3221.
16. Yusta, S. C. (2009). Different metaheuristic strategies to solve the feature selection
problem. Pattern Recognition Letters, 30(5), 525-534.
17. Ramos-Figueroa, O., Quiroz-Castellanos, M., Mezura-Montes, E., &amp; Schütze, O. (2020).
Metaheuristics to solve grouping problems: A review and a case study. Swarm and
Evolutionary Computation, 100643.
18. Santhanam, T., &amp; Padmavathi, M. S. (2015). Application of K-means and genetic
algorithms for dimension reduction by integrating SVM for diabetes diagnosis. Procedia
Computer Science, 47, 76-83.
19. Tran, B., Xue, B., &amp; Zhang, M. (2018). Variable-length particle swarm optimization for
feature selection on high-dimensional classification. IEEE Transactions on Evolutionary
Computation, 23(3), 473-487.
20. Thangavel, K., &amp; Pethalakshmi, A. (2009). Dimensionality reduction based on rough set
theory: A review. Applied Soft Computing, 9(1), 1-12.
21. Dash, M., &amp; Liu, H. (1997). Feature selection for classification. Intelligent data analysis,
1(3), 131-156.
22. Kwak, N., &amp; Choi, C. H. (2002). Input feature selection for classification problems. IEEE
transactions on neural networks, 13(1), 143-159.
23. Tang, J., Alelyani, S., &amp; Liu, H. (2014). Feature selection for classification: A review.
Data classification: Algorithms and applications, 37.
24. Xue, B., Zhang, M., &amp; Browne, W. N. (2012). Particle swarm optimization for feature
selection in classification: A multi-objective approach. IEEE transactions on cybernetics,
43(6), 1656-1671.
25. Inbarani, H. H., Bagyamathi, M., &amp; Azar, A. T. (2015). A novel hybrid feature selection
method based on rough set and improved harmony search. Neural Computing and
Applications, 26(8), 1859-1880.
26. Rouhi, A., &amp; Nezamabadi-pour, H. (2017, March). A hybrid feature selection approach
based on ensemble method for high-dimensional data. In 2017 2nd Conference on Swarm
Intelligence and Evolutionary Computation (CSIEC) (pp. 16-20). IEEE.
27. Abualigah, L. M., Khader, A. T., Al-Betar, M. A., &amp; Alomari, O. A. (2017). Text feature
selection with a robust weight scheme and dynamic dimension reduction to text document
clustering. Expert Systems with Applications, 84, 24-36.
28. Storn, R., &amp; Price, K. (1997). Differential evolution–a simple and efficient heuristic for
global optimization over continuous spaces. Journal of global optimization, 11(4), 341-359.
29. Mohamed, A. W., &amp; Mohamed, A. K. (2019). Adaptive guided differential evolution
algorithm with novel mutation for numerical optimization. International Journal of
Machine Learning and Cybernetics, 10(2), 253-277.
30. Balint Antal, Andras Hajdu: An ensemble-based system for automatic screening of
diabetic retinopathy, Knowledge-Based Systems 60 (April 2014), 20-27.
31. O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis
via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.
32. R. Alizadehsani, J. Habibi, M. J. Hosseini, H. Mashayekhi, R. Boghrati, A.
Ghandeharioun, et al., &#39;A data mining approach for diagnosis of coronary artery disease,&#39;
Computer Methods and Programs in Biomedicine, vol. 111, pp. 52-61, 2013/07/01/ 2013.
33. Johnson, B., 2013. High resolution urban land cover classification using a competitive
multi-scale object-based approach. Remote Sensing Letters, 4 (2), 131-140.
34. Arslan, F., &amp; KAHRAMAN, H. T. Yapay Zekâ Tabanlı Büyük Veri Yönetim
Aracı. Journal of Investigations on Engineering and Technology, 2(1), 8-21.
35. Yeşilbudak, M., Kahraman, H., &amp; Karacan, H. (2011). Veri madenciliğinde nesne
yönelimli birleştirici hiyerarşik kümeleme modeli. Gazi Üniversitesi Mühendislik-
Mimarlık Fakültesi Dergisi, 26(1).
36. Adak, M. F., &amp; Yurtay, N. (2013). Gini Algoritmasını Kullanarak Karar Ağacı
Oluşturmayı Sağlayan Bir Yazılımın Geliştirilmesi. Bilişim Teknolojileri Dergisi, 6(3), 1-6.
37. Lezki, Ş. (2014). Çok Kriterli Karar Verme Problemlerinde Karar Ağacı Kullanımı.
İktisadi Yenilik Dergisi, 2(1), 16-31.
38. Tokgöz, B. (2017). “Iş Gereksinimi Odaklı Test Senaryoları Üretim Modeli”, Necmettin Erbakan Üniversitesi Fen Bilimleri Enstitüsü Yüksek Lisans Tezi, 8-13.
39. Birbil, İ., &quot;Tahmin ve Çıkarım 4 - Boyut Küçültme&quot;,
http://www.veridefteri.com/2018/06/19/tahmin-ve-cikarim-4-boyut-kucultme/ (Son erişim
tarihi: 20.08. 2019)
40. Şimşek, H.K., &quot;Boyut Azaltma: Temel Bileşen Analizi&quot;, https://medium.com/deep-
learning-turkiye/boyut-azaltma-temel-bileşen-analizi-812fd2163bbf (Son erişim tarihi:
20.08. 2019)
41. Bingham, E., &amp; Mannila, H. (2001, August). Random projection in dimensionality
reduction: applications to image and text data. In Proceedings of the seventh ACM
SIGKDD international conference on Knowledge discovery and data mining (pp. 245-250). ACM.
42. Holland, J.H., 1975. &quot;Adaptation in natural and artificial systems: An introductory
analysis with applications to biology, control, and artificial intelligence&quot;. Q. Rev. Biol. 1,211. http://dx.doi.org/10.1086/418447.
43. Kahraman, H. T., Aras, S., Sönmez, Y., Güvenç, U., &amp; Gedikli, E. Analysis, Test and
Management of the Meta-Heuristic Searching Process: An Experimental Study on SOS.
Politeknik Dergisi.
44. Aras, S., Kahraman, H. T., Sönmez, Y., &amp; Güvenç, U. Meta-Sezgisel Arama
Algoritmalarının Test Edilmesi İçin Yeni Yöntemlerin Geliştirilmesi. İleri Teknoloji
Bilimleri Dergisi, 6(2), 1-8.
45. Murty, K. G.(2003). Optimization Models For Decision Making, vol. 1, Internet Edition,
Chapter1: Models for Decision Making, 1-18, 2003.
46. Dosoglu, M. K., Guvenc, U., Duman, S., Sonmez, Y., &amp; Kahraman, H. T. Symbiotic
organisms search optimization algorithm for economic/emission dispatch problem in
power systems. Neural Computing and Applications, 29(3), 721-737, 2018.
47. Baysal Y.A., Altaş İ.H., Gedikli E., &quot;Simbiyotik Organizmalar Arama Algoritması ile
Dağıtım Şebekelerinde Güç Kayıplarının Azaltılması&quot;, Akıllı Sistemlerde Yenilikler ve
Uygulamaları Sempozyumu (ASYU 2016), DÜZCE, TÜRKIYE, 29 Eylül - 1 Ekim 2016,
ss.295-299.
