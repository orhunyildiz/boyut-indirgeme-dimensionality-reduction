# boyut-indirgeme-dimensionality-reduction
*Sınıflandırma problemleri için meta-sezgisel boyut azaltma aracının tasarımı ve uygulaması (Bitirme Tezi)*

# İçindekiler
- [Özet](#özet)
1. Genel Bilgiler

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

# 1. Genel Bilgiler
  ## - 1.1 Giriş
