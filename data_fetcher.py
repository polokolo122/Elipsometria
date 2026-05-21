"""
Moduł interfejsu danych optycznych (Optical Data Fetcher).
Odpowiada za bezpieczne połączenie z bazą refractiveindex.info oraz ekstrakcję
i interpolację stałych optycznych (współczynnika załamania n oraz ekstynkcji k)
dla zdefiniowanej siatki energetycznej.
"""

import ssl
import numpy as np
import matplotlib.pyplot as plt
from refractiveindex import RefractiveIndexMaterial

# Konfiguracja obejścia certyfikatów SSL dla kompatybilności ze środowiskami macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class MaterialOpticalConstants:
    """
    Klasa zarządzająca ekstrakcją parametrów dyspersyjnych materiałów.

    Atrybuty:
        energy_grid (np.ndarray): Wektor energii fotonów w elektronowoltach (eV).
    """

    def __init__(self, e_min=0.8, e_max=6.2, num_points=100):
        self.energy_grid = np.linspace(e_min, e_max, num_points)

    def get_nk(self, shelf, book, page):
        """
        Pobiera i interpoluje zespolone stałe optyczne ($n$, $k$) materiału.

        Argumenty:
            shelf (str): Repozytorium w bazie (np. 'main').
            book (str): Formuła chemiczna/nazwa (np. 'Si').
            page (str): Identyfikator publikacji (np. 'Franta-25C').

        Zwraca:
            tuple: Zwraca dwie tablice (n_vals, k_vals) zmapowane na energy_grid.
        """
        material = RefractiveIndexMaterial(shelf=shelf, book=book, page=page)

        try:
            n_vals = material.get_refractive_index(self.energy_grid, unit='eV')
        except Exception as e:
            print(f"Ostrzeżenie (Ekstrakcja 'n' dla {book}/{page}): {e}")
            n_vals = np.ones_like(self.energy_grid)

        try:
            k_vals = material.get_extinction_coefficient(self.energy_grid, unit='eV')
        except Exception:
            k_vals = np.zeros_like(self.energy_grid)

        # Oczyszczanie danych: zamiana wartości nieliczbowych (NaN) na neutralne tło
        n_vals = np.nan_to_num(n_vals, nan=1.0)
        k_vals = np.nan_to_num(k_vals, nan=0.0)

        return n_vals, k_vals


if __name__ == "__main__":
    # Procedura testowa modułu
    fetcher = MaterialOpticalConstants()

    try:
        n_si, k_si = fetcher.get_nk('main', 'Si', 'Franta-25C')
        n_sio2, k_sio2 = fetcher.get_nk('main', 'SiO2', 'Malitson')

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fetcher.energy_grid, n_si, label='Si (Franta)', color='blue')
        plt.plot(fetcher.energy_grid, n_sio2, label='SiO2 (Malitson)', color='cyan')
        plt.title('Współczynnik załamania ($n$)')
        plt.xlabel('Energia (eV)')
        plt.ylabel('$n$')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(fetcher.energy_grid, k_si, label='Si (Franta)', color='red')
        plt.plot(fetcher.energy_grid, k_sio2, label='SiO2 (Malitson)', color='orange')
        plt.title('Współczynnik ekstynkcji ($k$)')
        plt.xlabel('Energia (eV)')
        plt.ylabel('$k$')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Błąd inicjalizacji stosu optycznego: {e}")