"""
Physics-Informed Neural Network (PINN) Validation.
Skrypt domykający pętlę weryfikacyjną. Wykorzystuje model ML do ewaluacji
ukrytych parametrów struktury (Problem Odwrotny), po czym zasila nimi bazowy
model fizyczny (Problem Prosty - Równania Fresnela/TMM) i rysuje residua wizualne.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random

from train_model import EllipsometryInverseNet
from data_fetcher import MaterialOpticalConstants
from dataset_generator import calculate_ellipsometry

MATERIALS_DB = {
    'SiO2': ('main', 'SiO2', 'Malitson'),
    'Si3N4': ('main', 'Si3N4', 'Luke'),
    'Al2O3': ('main', 'Al2O3', 'Franta'),
    'HfO2': ('main', 'HfO2', 'Franta'),
    'TiO2': ('main', 'TiO2', 'Franta'),
    'ZnO': ('main', 'ZnO', 'Aguilar'),
    'ITO': ('other', 'In2O3-SnO2', 'Konig'),
    'a-Si': ('main', 'Si', 'Franta-25C'),
    'ZrO2': ('main', 'ZrO2', 'Synowicki'),
    'TiN': ('main', 'TiN', 'Pfluger')
}


def validate_physics():
    """Generuje interfejs weryfikacji fotonicznej poprzez rekonstrukcję sygnału."""
    print("Inicjacja walidatora PINN...")
    le = joblib.load('label_encoder.pkl')
    scaler_reg = joblib.load('scaler_reg.pkl')
    scaler_psi = joblib.load('scaler_psi.pkl')
    scaler_delta = joblib.load('scaler_delta.pkl')

    df = pd.read_csv('ellipsometry_dataset.csv')

    # Wybór próbki weryfikacyjnej typu Blind-Test
    random_idx = random.randint(0, len(df) - 1)
    sample = df.iloc[random_idx]

    true_mat = sample['Material']
    true_thick = sample['Thickness_nm']
    true_rough = sample['Roughness_nm']

    psi_cols = [col for col in df.columns if col.startswith('Psi_')]
    delta_cols = [col for col in df.columns if col.startswith('Delta_')]

    true_psi = sample[psi_cols].values.astype(float)
    true_delta = sample[delta_cols].values.astype(float)

    print(f"\n--- WYLOSOWANA REFERENCJA ---")
    print(f"Struktura: {true_mat}")
    print(f"Wymiar Z:  {true_thick:.2f} nm")

    # ETAP 1: Dekodowanie Ukrytych Zmiennych (Estymator Sieciowy)
    print("\n[ETAP 1] Wyznaczanie parametrów przez model AI...")
    X_psi = scaler_psi.transform(true_psi.reshape(1, -1))
    X_delta = scaler_delta.transform(true_delta.reshape(1, -1))
    X_tensor = torch.tensor(np.stack((X_psi, X_delta), axis=1), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EllipsometryInverseNet(num_materials=len(le.classes_)).to(device)
    model.load_state_dict(torch.load('ellipsometry_model.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        pred_mat_logits, pred_mu, pred_var = model(X_tensor)

        pred_mat_idx = torch.argmax(pred_mat_logits, dim=1).cpu().numpy()[0]
        pred_mu = pred_mu.cpu().numpy()
        pred_var = pred_var.cpu().numpy()

    pred_mat = le.classes_[pred_mat_idx]
    pred_thick, pred_rough = scaler_reg.inverse_transform(pred_mu)[0]
    uncert_thick, uncert_rough = np.sqrt(pred_var)[0] * scaler_reg.data_range_

    print(f"Wykryty Ośrodek: {pred_mat}")
    print(f"Wymiar Wyznaczony: {pred_thick:.2f} nm (Sigma Ufności +/- {uncert_thick:.2f} nm)")

    # ETAP 2: Transformacja Fizyczna w Dziedzinie Energii (Forward Pass)
    print("\n[ETAP 2] Przepuszczenie wektora wyjściowego przez Równania TMM...")
    fetcher = MaterialOpticalConstants()
    wavelengths = 1239.84 / fetcher.energy_grid

    n_si, k_si = fetcher.get_nk('main', 'Si', 'Franta-25C')
    shelf, book, page = MATERIALS_DB[pred_mat]
    n_mat, k_mat = fetcher.get_nk(shelf, book, page)

    pred_psi, pred_delta = calculate_ellipsometry(
        n_si, k_si, n_mat, k_mat, pred_thick, pred_rough, wavelengths, angle_deg=70.0
    )

    # ETAP 3: Renderowanie Zgodności
    print("Renderowanie sprzężenia zwrotnego...")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fetcher.energy_grid, true_psi, 'b-', label='Analizat Ref (Prawda)', linewidth=4, alpha=0.5)
    plt.plot(fetcher.energy_grid, pred_psi, 'r--', label='Synteza z Parametrów AI', linewidth=2)
    plt.title(rf'Walidacja Zgodności Fazy $\Psi$ ({pred_mat})')
    plt.xlabel('Energia (eV)')
    plt.ylabel(r'$\Psi$ (stopnie)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fetcher.energy_grid, true_delta, 'b-', label='Analizat Ref (Prawda)', linewidth=4, alpha=0.5)
    plt.plot(fetcher.energy_grid, pred_delta, 'r--', label='Synteza z Parametrów AI', linewidth=2)
    plt.title(rf'Walidacja Zgodności Amplitudy $\Delta$ ({pred_mat})')
    plt.xlabel('Energia (eV)')
    plt.ylabel(r'$\Delta$ (stopnie)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_physics()
