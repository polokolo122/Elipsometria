"""
Zaawansowany Model Inferencyjny 1D-CNN (Probabilistic Deep Learning).
Sieć wyznacza klasy materiałowe oraz stosuje estymację opartą na
prawdopodobieństwie (Gaussian NLL), zwracając wartości $\mu$ wraz z przewidywaną
niepewnością $\sigma^2$ dla parametrów struktury. Skrypt wdraża pełen protokół MLOps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class EllipsometryDataset(Dataset):
    """Zarządzanie wektorami wejścia-wyjścia w formacie Tensorów."""
    def __init__(self, X_psi, X_delta, y_mat, y_reg):
        self.X = torch.tensor(np.stack((X_psi, X_delta), axis=1), dtype=torch.float32)
        self.y_mat = torch.tensor(y_mat, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)

    def __len__(self):
        return len(self.y_mat)

    def __getitem__(self, idx):
        return self.X[idx], self.y_mat[idx], self.y_reg[idx]


def load_and_preprocess_data(csv_file):
    """
    Wczytuje zbiór i zapobiega zjawisku Data Leakage poprzez rozdzielenie
    zbiorów przed dokonaniem procedury standaryzacji statystycznej (Fit/Transform).
    Ekstportuje skalery do artefaktów produkcyjnych (.pkl).
    """
    print("Inicjalizacja pipeline'u pre-processingu...")
    df = pd.read_csv(csv_file)

    le = LabelEncoder()
    y_mat = le.fit_transform(df['Material'])

    psi_cols = [col for col in df.columns if col.startswith('Psi_')]
    delta_cols = [col for col in df.columns if col.startswith('Delta_')]
    y_reg_raw = df[['Thickness_nm', 'Roughness_nm']].values

    # Mitygacja wycieku danych (Data Leakage): Podział przed trenowaniem skalerów
    X_p_tr, X_p_te, X_d_tr, X_d_te, y_m_tr, y_m_te, y_r_tr, y_r_te = train_test_split(
        df[psi_cols].values, df[delta_cols].values, y_mat, y_reg_raw,
        test_size=0.2, random_state=42
    )

    scaler_psi = StandardScaler().fit(X_p_tr)
    scaler_delta = StandardScaler().fit(X_d_tr)
    scaler_reg = MinMaxScaler(feature_range=(0, 1)).fit(y_r_tr)

    # Transformacja na ustalonych wariancjach
    X_p_tr = scaler_psi.transform(X_p_tr)
    X_p_te = scaler_psi.transform(X_p_te)
    X_d_tr = scaler_delta.transform(X_d_tr)
    X_d_te = scaler_delta.transform(X_d_te)
    y_r_tr = scaler_reg.transform(y_r_tr)
    y_r_te = scaler_reg.transform(y_r_te)

    # Serializacja skalerów dla środowiska ewaluacyjnego/produkcyjnego
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler_reg, 'scaler_reg.pkl')
    joblib.dump(scaler_psi, 'scaler_psi.pkl')
    joblib.dump(scaler_delta, 'scaler_delta.pkl')

    return (
        EllipsometryDataset(X_p_tr, X_d_tr, y_m_tr, y_r_tr),
        EllipsometryDataset(X_p_te, X_d_te, y_m_te, y_r_te),
        le
    )


class EllipsometryInverseNet(nn.Module):
    """
    Architektura głębokiej ekstrakcji cech topologicznych z rozgałęzieniem na
    prawdopodobieństwo klas (Softmax) oraz regresję probabilistyczną (Mu + Var).
    """
    def __init__(self, num_materials=10):
        super(EllipsometryInverseNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )

        self.material_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_materials)
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
            # 4 wyjścia: [Grubość_Mu, Szorstkość_Mu, Grubość_Var, Szorstkość_Var]
            nn.Linear(64, 4)
        )

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        mat_preds = self.material_head(features)

        reg_out = self.geometry_head(features)
        mu = reg_out[:, 0:2]

        # Wymuszenie fizycznej poprawności wariancji (Var > 0)
        var = F.softplus(reg_out[:, 2:4]) + 1e-6

        return mat_preds, mu, var


def train_model():
    """Główna pętla wykorzystująca metrykę NLL (Negative Log Likelihood)."""
    BATCH_SIZE = 256
    EPOCHS = 40
    LEARNING_RATE = 0.001

    train_data, test_data, le = load_and_preprocess_data('ellipsometry_dataset.csv')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Target obliczeniowy: {device}")

    model = EllipsometryInverseNet(num_materials=len(le.classes_)).to(device)

    criterion_mat = nn.CrossEntropyLoss()
    criterion_reg = nn.GaussianNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("\nRozpoczęcie treningu w ujęciu Bayesowskim (Gaussian NLL)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_mat_batch, y_reg_batch in train_loader:
            X_batch, y_mat_batch, y_reg_batch = X_batch.to(device), y_mat_batch.to(device), y_reg_batch.to(device)
            optimizer.zero_grad()

            pred_mat, pred_mu, pred_var = model(X_batch)
            loss = criterion_mat(pred_mat, y_mat_batch) + 2.0 * criterion_reg(pred_mu, y_reg_batch, pred_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct_mat = 0
        total_samples = 0

        with torch.no_grad():
            for X_test, y_mat_test, y_reg_test in test_loader:
                X_test, y_mat_test, y_reg_test = X_test.to(device), y_mat_test.to(device), y_reg_test.to(device)
                p_mat, p_mu, p_var = model(X_test)

                val_loss += (criterion_mat(p_mat, y_mat_test) + 5.0 * criterion_reg(p_mu, y_reg_test, p_var)).item()
                correct_mat += (torch.argmax(p_mat, 1) == y_mat_test).sum().item()
                total_samples += y_mat_test.size(0)

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoka [{epoch+1:02d}/{EPOCHS}] | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {100*correct_mat/total_samples:.2f}%")

    torch.save(model.state_dict(), "ellipsometry_model.pth")
    print("Zapisano artefakt: ellipsometry_model.pth")


if __name__ == "__main__":
    train_model()