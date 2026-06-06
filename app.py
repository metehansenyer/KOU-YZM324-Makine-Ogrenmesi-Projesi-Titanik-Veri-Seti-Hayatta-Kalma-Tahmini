"""
Titanic Hayatta Kalma Tahmin Arayüzü
--------------------------------------
Çalıştırmak için:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle — her çalışma dizininden import çalışsın
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import streamlit as st

from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.modeling import get_models
from src.preprocessing import encode


# ---------------------------------------------------------------------------
# Kaynakların yüklenmesi (tek sefer — cache_resource ile)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """
    Eğitim pipeline'ındaki scaler'ı birebir yeniden üretir ve
    kaydedilmiş modelleri yükler.

    NOT: modeling.py:split_data scaler'ı kaydetmediği için burada
    aynı parametrelerle (test_size, random_state, stratify) fit ediyoruz.
    """
    df_clean = pd.read_csv(DATA_PROCESSED / "titanic_clean.csv")
    df_enc = encode(df_clean)

    X = df_enc.drop(columns=["Survived"])
    y = df_enc["Survived"]
    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_train)
    feature_columns = X_train.columns

    # modeling._model_path ile aynı adlandırma mantığı
    models = {
        name: joblib.load(
            MODELS_DIR / f"{name.lower().replace(' ', '_').replace('-', '_')}.joblib"
        )
        for name in get_models()
    }

    return scaler, feature_columns, models


# ---------------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Titanic Hayatta Kalma Tahmini",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic — Hayatta Kalma Tahmini")
st.write(
    "Yolcu bilgilerini girin, seçtiğiniz makine öğrenmesi modeli "
    "hayatta kalma olasılığınızı tahmin etsin."
)

scaler, feature_columns, models = load_artifacts()

# ---------------------------------------------------------------------------
# Kullanıcı girdileri
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Yolcu Bilgileri")

col1, col2 = st.columns(2)

with col1:
    cinsiyet_label = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    cinsiyet = "male" if cinsiyet_label == "Erkek" else "female"

    yas = st.slider("Yaş", min_value=1, max_value=80, value=30)

    sinif = st.selectbox(
        "Yolcu Sınıfı",
        [1, 2, 3],
        format_func=lambda x: f"{x}. Sınıf",
    )

    ucret = st.slider("Bilet Ücreti ($)", min_value=0, max_value=500, value=50)

    liman = st.selectbox(
        "Biniş Limanı",
        ["Cherbourg", "Queenstown", "Southampton"],
    )

with col2:
    kardes_es = st.slider("Kardeş / Eş Sayısı", min_value=0, max_value=8, value=0)

    ebeveyn_cocuk = st.slider("Ebeveyn / Çocuk Sayısı", min_value=0, max_value=6, value=0)

    _unvan_etiket = {
        "Bay (Mr)": "Mr",
        "Hanım / Evli (Mrs)": "Mrs",
        "Hanım / Bekar (Miss)": "Miss",
        "Genç Efendi (Master)": "Master",
        "Doktor (Dr)": "Dr",
        "Diğer": "Other",
    }
    unvan_label = st.selectbox("Unvan", list(_unvan_etiket.keys()))
    unvan = _unvan_etiket[unvan_label]

    _guverte_etiket = {
        "Bilinmiyor": "Unknown",
        "A": "A", "B": "B", "C": "C", "D": "D",
        "E": "E", "F": "F", "G": "G", "T": "T",
    }
    guverte_label = st.selectbox("Güverte", list(_guverte_etiket.keys()))
    guverte = _guverte_etiket[guverte_label]

st.divider()
st.subheader("Model Seçimi")

model_adi = st.radio(
    "Tahmin için kullanılacak model:",
    list(models.keys()),
    horizontal=True,
)

# ---------------------------------------------------------------------------
# Tahmin
# ---------------------------------------------------------------------------

st.divider()

if st.button("🔮 Tahmin Et", use_container_width=True, type="primary"):

    aile_boyutu = kardes_es + ebeveyn_cocuk + 1
    tek_basina = int(aile_boyutu == 1)

    # encode() fonksiyonunun beklediği sütun yapısıyla ham DataFrame
    row = pd.DataFrame([{
        "PassengerClass":   sinif,
        "Sex":              cinsiyet,
        "Age":              float(yas),
        "SiblingsSpouses":  kardes_es,
        "ParentsChildren":  ebeveyn_cocuk,
        "TicketFare":       float(ucret),
        "Title":            unvan,
        "Deck":             guverte,
        "FamilySize":       aile_boyutu,
        "IsAlone":          tek_basina,
        "Port":             liman,
        # encode() drop etmediği için Survived sütunu olmamalı;
        # load_artifacts'ta df_clean'den encode çağrıldığında Survived vardı
        # ama burada sadece girdi satırı var, yoksa encode hata vermez.
    }])

    # src.preprocessing.encode ile aynı dönüşüm
    encoded = encode(row)

    # Eğitimde oluşmayan dummy sütunları 0 ile doldur, sırayı uydur
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    # Ölçekleme (eğitimle aynı scaler) — DataFrame olarak sarmalayarak
    # feature name uyarılarını önle
    scaled = pd.DataFrame(
        scaler.transform(encoded),
        columns=feature_columns,
    )

    model = models[model_adi]
    proba = model.predict_proba(scaled)[0, 1]
    hayatta_kalir = proba >= 0.5

    # Sonuç gösterimi
    st.subheader("Tahmin Sonucu")
    st.caption(f"Model: **{model_adi}**")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(
            label="Hayatta Kalma Olasılığı",
            value=f"%{proba * 100:.1f}",
        )

    with col_res2:
        st.progress(float(proba))
        if hayatta_kalir:
            st.success("✅ Bu yolcu **hayatta kalırdı**.")
        else:
            st.error("❌ Bu yolcu **hayatta kalamazdı**.")

    with st.expander("Teknik Detaylar"):
        st.write("**Encode edilmiş & ölçeklenmiş girdi:**")
        st.dataframe(scaled.T.rename(columns={0: "Değer"}), use_container_width=True)
