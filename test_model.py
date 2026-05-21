"""
Moduł walidacyjny. Wykonuje inferencję (problem odwrotny) na modelu wczytanym
z dysku i oblicza globalne metryki jakości modelu (Mean Absolute Error).
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from train_model import EllipsometryInverseNet


def evaluate_model():
    """
    Wczytuje ukryty zbiór danych, odtwarza struktury skalujące
    oraz ewaluuje predykcje regresyjne i klasyfikacyjne modelu.
    """
    print("Odtwarzanie środowiska estymacji...")
    df = pd.read_csv('ellipsometry_dataset.csv')

    le = LabelEncoder()
    df['Material_Idx'] = le.fit_transform(df['Material'])

    scaler_reg = MinMaxScaler(feature_range=(0, 1))
    scaler_reg.fit(df[['Thickness_nm', 'Roughness_nm']].values)

    psi_cols = [col for col in df.columns if col.startswith('Psi_')]
    delta_cols = [col for col in df.columns if col.startswith('Delta_')]

    X_psi = StandardScaler().fit_transform(df[psi_cols].values)
    X_delta = StandardScaler().fit_transform(df[delta_cols].values)

    _, X_test_p, _, X_test_d, _, y_test_mat, _, y_test_reg = train_test_split(
        X_psi, X_delta, df['Material_Idx'].values, df[['Thickness_nm', 'Roughness_nm']].values,
        test_size=0.2, random_state=42
    )

    X_test_tensor = torch.tensor(np.stack((X_test_p, X_test_d), axis=1), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EllipsometryInverseNet(num_materials=len(le.classes_)).to(device)
    model.load_state_dict(torch.load('ellipsometry_model.pth', map_location=device))
    model.eval()

    print(f"Ewaluacja {len(X_test_tensor)} wektorów cech...")

    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        pred_mat_logits, pred_reg_scaled = model(X_test_tensor)

        pred_mat_idx = torch.argmax(pred_mat_logits, dim=1).cpu().numpy()
        pred_reg_scaled = pred_reg_scaled.cpu().numpy()

    pred_reg_nm = scaler_reg.inverse_transform(pred_reg_scaled)

    # Wizualizacja błędu bezwzględnego w dziedzinie fizycznej (nm)
    errors_thickness = np.abs(y_test_reg[:, 0] - pred_reg_nm[:, 0])
    mae_thick = np.mean(errors_thickness)
    print(f"\n[METRYKA] Średni Błąd Bezwzględny (MAE) dla grubości: {mae_thick:.2f} nm")

    # Wykres korelacji
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_reg[:, 0], pred_reg_nm[:, 0], alpha=0.3, color='blue', s=10)
    plt.plot([10, 800], [10, 800], color='red', linestyle='--', linewidth=2, label='Linia referencyjna (Y=X)')
    plt.title(f'Korelacja Predykcji 1D-CNN\nMAE: {mae_thick:.2f} nm')
    plt.xlabel('Grubość referencyjna (nm)')
    plt.ylabel('Grubość estymowana (nm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()