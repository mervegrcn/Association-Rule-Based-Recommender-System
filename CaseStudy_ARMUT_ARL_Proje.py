
# Recommendation Systems: ARL (Association Rule Learning)

#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_csv("/Users/mervegurcan/PycharmProjects/pythonProject/DATASETS/armut_data.csv")
df = df_.copy()

# Veri ön incelemesi yapma:
def check_dataframe(df, row_num=5):
    print("*************** Dataset Shape ***************")
    print("No. of Rows:", df.shape[0], "\nNo. of Columns:", df.shape[1])
    print("*************** Dataset Information ***************")
    print(df.info())
    print("*************** Types of Columns ***************")
    print(df.dtypes)
    print(f"*************** First {row_num} Rows ***************")
    print(df.head(row_num))
    print(f"*************** Last {row_num} Rows ***************")
    print(df.tail(row_num))
    print("*************** Summary Statistics of The Dataset ***************")
    print(df.describe().T)
    print("*************** Dataset Missing Values Analysis ***************")
    print(df.isnull().sum())
    print("*************** Dataset Duplicates ***************")
    print(df[df.duplicated() == True])


check_dataframe(df)

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

def prep_data (df):
    df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

    df['New_Date'] = pd.to_datetime(df['CreateDate']).dt.strftime('%Y-%m')
    df['BasketId'] = df['UserId'].astype(str) + '_' + df['New_Date']

    return df

prep_data(df)

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.
# eksik değerlerde 0, dolularda 1 yazmasını istiyoruz:

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

def prep_for_arl_recommender(df):

    apriori_df = df.groupby(['BasketId', 'Service']).agg({"CategoryId": "count"}).unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

# Adım 2: Birliktelik kurallarını oluşturunuz.

    # Apriori yöntemi ile olası tüm ürün birlikteliklerinin support değerlerini buluyoruz.

    frequent_itemsets = apriori(apriori_df,
                                min_support=0.01,
                                use_colnames=True)

    frequent_itemsets.sort_values("support", ascending=False)

    rules = association_rules(frequent_itemsets,
                              metric="support",
                              min_threshold=0.01)

    return rules

rules = prep_for_arl_recommender(df)

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, service, rec_count=1):
    # Sıralama kuralı belirleniyor:
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    # Filter the rules where the antecedent contains the given service
    filtered_rules = sorted_rules[sorted_rules["antecedents"].apply(lambda x: service in x)]

    # Öneri listesi oluşturma:
    recommendation_list = [', '.join(rule) for rule in filtered_rules["consequents"].apply(list).tolist()[:rec_count]]

    print("Other recommended service(s) for the service you entered: ", recommendation_list)

arl_recommender(rules, '2_0')

