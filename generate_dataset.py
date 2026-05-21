"""
Generator macierzy danych (Dataset Builder).
Wykorzystuje model sprzężony do masowej generacji widm z zachowaniem domenowych
limitów fizycznych głębokości wnikania dla ośrodków silnie absorbujących.
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_fetcher import MaterialOpticalConstants
from dataset_generator import calculate_ellipsometry

# Konfiguracja indeksu badanych materiałów
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

SAMPLES_PER_MATERIAL = 10000

# Redukcja obszaru wieloznaczności (Fringe Ambiguity Limit)
# Nałożenie granic fizycznej widoczności podłoża (Penetration Depth)
MAX_THICKNESS = {
    'TiN': 150.0,
    'a-Si': 150.0
}


def generate_database():
    """Generuje instancje danych i serializuje wynikowy tensor do pliku CSV."""
    fetcher = MaterialOpticalConstants()
    wavelengths = 1239.84 / fetcher.energy_grid

    print("Inicjalizacja środowiska i ekstrakcja stałych optycznych...")
    n_si, k_si = fetcher.get_nk('main', 'Si', 'Franta-25C')

    materials_nk = {}
    for mat_name, (shelf, book, page) in MATERIALS_DB.items():
        materials_nk[mat_name] = fetcher.get_nk(shelf, book, page)

    dataset_rows = []
    print(f"\nGeneracja (Wolumen: {SAMPLES_PER_MATERIAL} próbek / klasę)")

    for mat_name in MATERIALS_DB.keys():
        n_mat, k_mat = materials_nk[mat_name]
        limit_grubosci = MAX_THICKNESS.get(mat_name, 800.0)

        for _ in tqdm(range(SAMPLES_PER_MATERIAL), desc=f"Synteza: {mat_name}"):
            thickness = random.uniform(10.0, limit_grubosci)
            roughness = random.uniform(0.0, 10.0)

            psi, delta = calculate_ellipsometry(
                n_si, k_si, n_mat, k_mat, thickness, roughness, wavelengths, angle_deg=70.0
            )

            row_data = {
                'Material': mat_name,
                'Thickness_nm': round(thickness, 2),
                'Roughness_nm': round(roughness, 2)
            }

            for i, p_val in enumerate(psi):
                row_data[f'Psi_{i}'] = p_val
            for i, d_val in enumerate(delta):
                row_data[f'Delta_{i}'] = d_val

            dataset_rows.append(row_data)

    print("\nSerializacja do DataFrame...")
    df = pd.DataFrame(dataset_rows)
    csv_filename = "ellipsometry_dataset.csv"
    df.to_csv(csv_filename, index=False)

    size_mb = os.path.getsize(csv_filename) / (1024 * 1024)
    print(f"Eksport zakończony: {csv_filename} ({size_mb:.2f} MB, Ilość: {len(df)})")


if __name__ == "__main__":
    generate_database()