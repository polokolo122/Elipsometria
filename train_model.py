"""
Główny moduł uczenia maszynowego (Deep Learning Trainer).
Definiuje architekturę sieci 1D-CNN z dwoma wyjściami (Multi-Task Learning)
do jednoczesnej klasyfikacji materiału oraz regresji parametrów geometrycznych.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class EllipsometryDataset(Dataset):
    """Integracja wektorów numpy z infrastrukturą tensorów PyTorch."""

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
    Inicjalizuje pipeline przetwarzania danych: mapowanie klas,
    skalowanie MinMax dla wartości docelowych oraz standaryzację Z-score dla sygnałów.
    """
    df = pd.read_csv(csv_file)

    le = LabelEncoder()
    y_mat = le.fit_transform(df['Material'])

    y_reg_raw = df[['Thickness_nm', 'Roughness_nm']].values
    scaler_reg = MinMaxScaler(feature_range=(0, 1))
    y_reg = scaler_reg.fit_transform(y_reg_raw)

    psi_cols = [col for col in df.columns if col.startswith('Psi_')]
    delta_cols = [col for col in df.columns if col.startswith('Delta_')]

    X_psi = StandardScaler().fit_transform(df[psi_cols].values)
    X_delta = StandardScaler().fit_transform(df[delta_cols].values)

    X_p_tr, X_p_te, X_d_tr, X_d_te, y_m_tr, y_m_te, y_r_tr, y_r_te = train_test_split(
        X_psi, X_delta, y_mat, y_reg, test_size=0.2, random_state=42
    )

    train_ds = EllipsometryDataset(X_p_tr, X_d_tr, y_m_tr, y_r_tr)
    test_ds = EllipsometryDataset(X_p_te, X_d_te, y_m_te, y_r_te)

    return train_ds, test_ds, le, scaler_reg


class EllipsometryInverseNet(nn.Module):
    """
    Architektura jednowymiarowej sieci konwolucyjnej (1D-CNN).
    Składa się z bloku ekstrakcji cech topologicznych widma oraz dwóch
    głowic dedykowanych zadaniom klasyfikacji i regresji.
    """

    def __init__(self, num_materials=10):
        super(EllipsometryInverseNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.material_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_materials)
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.material_head(features), self.geometry_head(features)


def train_model():
    """
    Główna pętla sterująca treningiem modelu. Wykorzystuje ważoną sumę
    funkcji straty (CrossEntropy oraz L1Loss) optymalizowaną algorytmem Adam.
    """
    BATCH_SIZE = 256
    EPOCHS = 40
    LEARNING_RATE = 0.001

    train_data, test_data, material_encoder, _ = load_and_preprocess_data('ellipsometry_dataset.csv')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inicjalizacja środowiska obliczeniowego: {device}")

    model = EllipsometryInverseNet(num_materials=len(material_encoder.classes_)).to(device)

    criterion_mat = nn.CrossEntropyLoss()
    criterion_reg = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_mat_batch, y_reg_batch in train_loader:
            X_batch, y_mat_batch, y_reg_batch = X_batch.to(device), y_mat_batch.to(device), y_reg_batch.to(device)

            optimizer.zero_grad()
            pred_mat, pred_reg = model(X_batch)

            # Dominująca waga błędu regresji wymusza priorytetyzację geometrii struktury
            loss = criterion_mat(pred_mat, y_mat_batch) + (10.0 * criterion_reg(pred_reg, y_reg_batch))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Faza walidacyjna
        model.eval()
        val_loss = 0.0
        correct_mat = 0

        with torch.no_grad():
            for X_test, y_mat_test, y_reg_test in test_loader:
                X_test, y_mat_test, y_reg_test = X_test.to(device), y_mat_test.to(device), y_reg_test.to(device)
                p_mat, p_reg = model(X_test)

                v_loss_m = criterion_mat(p_mat, y_mat_test)
                v_loss_r = criterion_reg(p_reg, y_reg_test)
                val_loss += (v_loss_m + (10.0 * v_loss_r)).item()

                _, predicted_classes = torch.max(p_mat, 1)
                correct_mat += (predicted_classes == y_mat_test).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        accuracy = 100.0 * correct_mat / len(test_loader.dataset)
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

    torch.save(model.state_dict(), "ellipsometry_model.pth")


if __name__ == "__main__":
    train_model()