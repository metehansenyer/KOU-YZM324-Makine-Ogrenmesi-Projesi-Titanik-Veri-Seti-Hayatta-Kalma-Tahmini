# KOÜ Yazılım Müh. Makine Öğrenmesi Temelleri Dersi Projesi
Kocaeli Üniversitesi Mühendislik Fakültesi Yazılım Mühendisliği 25-26 Makine Öğrenmesi Temelleri Projesi GitHub sayfası. Titanic Hayatta Kalma Tahmini.

# İçerik

- [KOÜ Yazılım Müh. Makine Öğrenmesi Temelleri Dersi Projesi](#koü-yazılım-müh-makine-öğrenmesi-temelleri-dersi-projesi)
- [İçerik](#i̇çerik)
  - [Kullanılan Araçlar](#kullanılan-araçlar)
  - [Amaç](#amaç)
  - [Karşılanan Beklentiler](#karşılanan-beklentiler)
  - [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
    - [Yöntem 1 — CLI](#yöntem-1--cli)
    - [Yöntem 2 — Jupyter Notebook](#yöntem-2--jupyter-notebook)

## Kullanılan Araçlar

<p align="center">
  <a href="https://www.python.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a>
  <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" width="40" height="40"/> </a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit-learn" width="40" height="40"/> </a>
  <a href="https://jupyter.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg" alt="jupyter" width="40" height="40"/> </a>
</p>

- Projenin tamamı Python 3.13 dili ile geliştirilmiştir.
- Veri işleme için pandas ve numpy kütüphaneleri kullanılmıştır.
- Makine öğrenmesi modelleri scikit-learn ile eğitilmiştir.
- Görselleştirmeler matplotlib ve seaborn ile üretilmiştir.
- Anlatımlı analiz Jupyter Notebook üzerinden sunulmaktadır.

| Kullanılan Araç |                                Tavsiye Linkler                                 |
| :-------------: | :----------------------------------------------------------------------------: |
|     Python      |               [Resmi Dokümantasyon](https://docs.python.org/3/)                |
|     pandas      |            [pandas Dokümantasyonu](https://pandas.pydata.org/docs/)            |
|  scikit-learn   | [scikit-learn Dokümantasyonu](https://scikit-learn.org/stable/user_guide.html) |
|   matplotlib    |  [matplotlib Dokümantasyonu](https://matplotlib.org/stable/users/index.html)   |
|     seaborn     |       [seaborn Dokümantasyonu](https://seaborn.pydata.org/tutorial.html)       |
|    Veri Seti    |       [Kaggle — Titanic Dataset](https://www.kaggle.com/c/titanic/data)        |

## Amaç

YZM324 dersi kapsamında, bugüne kadar öğrendiğimiz makine öğrenmesi bilgileri ve Python dili kullanılarak Titanic yolcularının hayatta kalıp kalmadığını tahmin eden bir ikili sınıflandırma projesi geliştirilmesi hedeflenmektedir. Proje; veri temizleme, özellik mühendisliği, model eğitimi ve karşılaştırmalı değerlendirme adımlarını kapsayarak öğrencilerin uçtan uca bir makine öğrenmesi pipeline'ı kurma becerilerini pekiştirmesini amaçlamaktadır.

## Karşılanan Beklentiler

|        Beklenti         | Durum |                                        Detay                                        |
| :---------------------: | :---: | :---------------------------------------------------------------------------------: |
|     Veri Hazırlama      |   ✅   |         Tüm özellik mühendisliği ve eksik değer stratejileri uygulanmıştır.         |
|     EDA Grafikleri      |   ✅   |                3 EDA grafiği headless Agg backend ile üretilmiştir.                 |
|      Model Eğitimi      |   ✅   |            4 model aynı bölme üzerinde eğitilmiş ve karşılaştırılmıştır.            |
| Değerlendirme Çıktıları |   ✅   |     9 PNG (fig1–fig5 + 4 karışıklık matrisi) ve model_results.csv üretilmiştir.     |
|     CLI Çalıştırma      |   ✅   |      `python main.py` komutu pipeline'ı baştan sona hatasız çalıştırmaktadır.       |
|    Jupyter Notebook     |   ✅   | Notebook adım adım açıklamalar içermekte ve çıktıları gömülü şekilde çalışmaktadır. |
|  Teknik Gereklilikler   |   ✅   | Veri sızıntısı önlendi; tüm stokastik adımlar sabitlendi; mantık src/'de toplandı.  |

## Kurulum ve Çalıştırma

```bash
git clone https://github.com/metehansenyer/KOU-YZM324-MachineLearning-TitanicProject.git
cd KOU-YZM324-MachineLearning-TitanicProject
```

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Yöntem 1 — CLI

```bash
python main.py
```

### Yöntem 2 — Jupyter Notebook

```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

> [!NOTE]
> Her iki yöntem de `results/figures/` ve `data/processed/` klasörlerini otomatik olarak oluşturur.
> Klasörlerin önceden mevcut olması gerekmiyor.

> [!IMPORTANT]
> `main.py` proje kök dizininden çalıştırılmalıdır.
> Notebook ise `notebooks/` klasöründen açılmalıdır; `sys.path` ayarı otomatik yapılmaktadır.
