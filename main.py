from src.data_loader import load_data
from src.features import engineer_features
from src.preprocessing import handle_missing, rename_and_drop, encode
from src.eda import run_eda
from src.modeling import split_data, train_and_evaluate
from src.evaluation import compare_models


def main():
    print("=== Adım 1: Veri Yükleme ===")
    df = load_data()

    print("\n=== Adım 2: Keşifsel Veri Analizi (EDA) ===")
    run_eda(df)

    print("\n=== Adım 3: Özellik Mühendisliği ===")
    df = engineer_features(df)

    print("\n=== Adım 4: Eksik Değerlerin Giderilmesi ===")
    df = handle_missing(df)

    print("\n=== Adım 5-6: Yeniden Adlandırma, Temizleme ve Kaydetme ===")
    df_clean = rename_and_drop(df, save=True)

    print("\n=== Adım 7: Kodlama ===")
    df_enc = encode(df_clean)

    print("\n=== Adım 8: Veri Bölme ve Ölçekleme ===")
    X_train, X_test, y_train, y_test = split_data(df_enc)
    print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")

    print("\n=== Adım 9-10: Model Eğitimi ve Değerlendirmesi ===")
    results_df, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n=== Adım 11: Model Karşılaştırması ===")
    compare_models(results_df)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
