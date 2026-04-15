# Face-Verification-MobileNetV2
from deepface import DeepFace

# 1. Veritabanımız (13 bin resmin bulunduğu ana klasör)
db_path = r"C:\Users\cagat\OneDrive\Desktop\makine_ogrenmesi\lfw-deepfunneled"

# 2. Aramak istediğimiz (sorgu) resim 
# NOT: BURADAKİ DOSYA YOLUNU DEĞİŞTİREREK İSTEDİĞİN HERHANGİ BİR KİŞİYİ ARATABİLİRSİN!
img_path = r"C:\Users\cagat\OneDrive\Desktop\2e369985-1270-4805-86e1-fde7a7f97629.jpg"
# Örnek başka bir kişi: r"C:\Users\cagat\OneDrive\Desktop\makine_ogrenmesi\lfw-deepfunneled\Colin_Powell\Colin_Powell_0001.jpg"

print(f"Arama başlatıldı, veri tabanı taranıyor...\n")

# 3. Yüz Tanıma ve Arama İşlemi (1:N Identification)
# model_name: Akademik olarak en güçlü modellerden VGG-Face kullanıldı
# distance_metric: Işık değişimlerine en dayanıklı olan Kosinüs (cosine) benzerliği seçildi
# enforce_detection=False: LFW veri setindeki yüzü net olmayan bozuk fotoğraflarda kodun çökmemesi için eklendi
results = DeepFace.find(
    img_path = img_path, 
    db_path = db_path, 
    model_name = 'VGG-Face', 
    distance_metric = 'cosine', 
    enforce_detection = False
)

print("--- EN İYİ 5 EŞLEŞME ---")

# 4. Sonuçları Ekrana Yazdırma
if len(results) > 0 and not results[0].empty:
    df = results[0] # Sonuçları bir tablo (dataframe) olarak alıyoruz
    
    # head(5) metodu ile en çok benzeyen ilk 5 sonucu sırayla ekrana bastırıyoruz
    for index, row in df.head(5).iterrows():
        print(f"{index+1}. Sonuç: {row['identity']}")
        print(f"   Kosinüs Mesafesi: {row['distance']:.4f}\n")
else:
    print("Üzgünüm, veri tabanında benzer biri bulunamadı.")
