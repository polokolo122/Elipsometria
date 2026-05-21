"""
Ewaluacja rozkładu Gaussa na zbiorze weryfikacyjnym.
Odtwarza zapisaną pamięć środowiska estymacyjnego i wyciąga fizyczną niepewność
($\sigma$) estymacji dla każdej obserwacji testowej.
"""

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from train_model import EllipsometryInverseNet


def evaluate_model():
    """Moduł testowania jakości kalibracji sieci (MACE/NLL)."""
    print("Ładowanie artefaktów MLOps (.pkl)...")
    le = joblib.load('label_encoder.pkl')
    scaler_reg = joblib.load('scaler_reg.pkl')
    scaler_psi = joblib.load('scaler_psi.pkl')
    scaler_delta = joblib.load('scaler_delta.pkl')

    df = pd.read_csv('ellipsometry_dataset.csv')
    psi_cols = [col for col in df.columns if col.startswith('Psi_')]
    delta_cols = [col for col in df.columns if col.startswith('Delta_')]

    y_reg_raw = df[['Thickness_nm', 'Roughness_nm']].values
    y_mat = le.transform(df['Material'])

    # Bezpieczna izolacja przestrzeni testowej
    X_p_tr, X_p_te, X_d_tr, X_d_te, _, y_test_mat, _, y_r_te = train_test_split(
        df[psi_cols].values, df[delta_cols].values, y_mat, y_reg_raw,
        test_size=0.2, random_state=42
    )

    X_test_p = scaler_psi.transform(X_p_te)
    X_test_d = scaler_delta.transform(X_d_te)
    y_test_reg = scaler_reg.transform(y_r_te)

    X_test_tensor = torch.tensor(np.stack((X_test_p, X_test_d), axis=1), dtype=torch.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EllipsometryInverseNet(num_materials=len(le.classes_)).to(device)
    model.load_state_dict(torch.load('ellipsometry_model.pth', map_location=device))
    model.eval()

    print(f"Model wczytany. Inferencja statystyczna: {len(X_test_tensor)} wektorów.\n")

    with torch.no_grad():
        pred_mat_logits, pred_mu, pred_var = model(X_test_tensor.to(device))

        pred_mat_idx = torch.argmax(pred_mat_logits, dim=1).cpu().numpy()
        pred_mu = pred_mu.cpu().numpy()
        pred_var = pred_var.cpu().numpy()

    # Ekstrakcja wielkości fizycznych (nm)
    pred_mu_nm = scaler_reg.inverse_transform(pred_mu)
    y_test_reg_nm = scaler_reg.inverse_transform(y_test_reg)

    # Przekształcenie wariancji topologicznej w odchylenie standardowe w przestrzeni fizycznej
    sigma_nm = np.sqrt(pred_var) * scaler_reg.data_range_

    print("-" * 75)
    print("WERYFIKACJA STANU (OSTATECZNE PREDYKCJE WRAZ Z ODCHYLENIEM)")
    print("-" * 75)

    for idx in np.random.choice(len(X_test_tensor), 5, replace=False):
        true_mat = le.classes_[y_test_mat[idx]]
        pred_mat = le.classes_[pred_mat_idx[idx]]

        true_thick, _ = y_test_reg_nm[idx]
        pred_thick, _ = pred_mu_nm[idx]
        uncert_thick, _ = sigma_nm[idx]
        err_thick = abs(true_thick - pred_thick)

        print(f"Materiał: Referencja = {true_mat:5} | Estymata = {pred_mat:5} {'✅' if true_mat == pred_mat else '❌'}")
        print(
            f"Grubość:  Referencja = {true_thick:6.2f} nm | Model: {pred_thick:6.2f} +/- {uncert_thick:.2f} nm | AbsError: {err_thick:.2f} nm")
        print("-" * 75)

    mae = np.mean(np.abs(y_test_reg_nm[:, 0] - pred_mu_nm[:, 0]))
    mat_acc = np.mean(pred_mat_idx == y_test_mat) * 100
    within_1sigma = np.abs(y_test_reg_nm[:, 0] - pred_mu_nm[:, 0]) < sigma_nm[:, 0]

    print(f"\n[METRYKI GLOBALNE]")
    print(f"Średni MAE Grubości: {mae:.2f} nm")
    print(f"Globalna Celność Klasyfikacji: {mat_acc:.2f}%")
    print(f"Stopień pokrycia ufności (w granicach 1 Sigma): {within_1sigma.mean() * 100:.1f}%\n")

    print("[WYNIKI REGRESJI Z PODZIAŁEM NA KLASY MATERIAŁOWE]")
    for mat in le.classes_:
        mask = y_test_mat == le.transform([mat])[0]
        if np.sum(mask) > 0:
            mae_mat = np.mean(np.abs(y_test_reg_nm[mask, 0] - pred_mu_nm[mask, 0]))
            sigma_mat = np.mean(sigma_nm[mask, 0])
            print(f"{mat:6}: Średni MAE = {mae_mat:6.2f} nm | Deklarowana Sigma = {sigma_mat:5.2f} nm")


if __name__ == "__main__":
    evaluate_model()