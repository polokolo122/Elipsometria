"""
Fizyczny symulator elipsometryczny (Forward Model).
Implementuje metodę macierzy przejścia (Transfer Matrix Method - TMM) dla
wielowarstwowych struktur cienkowarstwowych. Wykorzystuje przybliżenie ośrodka
efektywnego Bruggemana (EMA) do modelowania chropowatości powierzchni.
"""

import numpy as np
import matplotlib.pyplot as plt
from tmm import coh_tmm
from data_fetcher import MaterialOpticalConstants


def bruggeman_ema(n_material, n_ambient=1.0, f_material=0.5):
    """
    Oblicza efektywny współczynnik załamania dla warstwy chropowatej z użyciem
    modelu Bruggemana (Effective Medium Approximation).
    Rozwiązuje równanie:
    $f \\frac{\\epsilon_m - \\epsilon_{eff}}{\\epsilon_m + 2\\epsilon_{eff}} + (1-f) \\frac{\\epsilon_a - \\epsilon_{eff}}{\\epsilon_a + 2\\epsilon_{eff}} = 0$

    Argumenty:
        n_material (complex): Zespolony współczynnik załamania materiału bazy.
        n_ambient (float): Współczynnik załamania otoczenia (domyślnie 1.0 dla próżni/powietrza).
        f_material (float): Ułamek objętościowy materiału (0.0 - 1.0).

    Zwraca:
        complex: Zespolony współczynnik załamania ośrodka efektywnego ($N_{eff}$).
    """
    e_m = n_material ** 2
    e_a = n_ambient ** 2

    b = (3 * f_material - 1) * e_m + (2 - 3 * f_material) * e_a
    delta = b ** 2 + 8 * e_m * e_a
    e_eff = (b + np.sqrt(delta)) / 4.0

    return np.sqrt(e_eff)


def calculate_ellipsometry(n_si, k_si, n_mat, k_mat, thickness_nm, roughness_nm, wavelengths_nm, angle_deg=70.0):
    """
    Oblicza teoretyczne widma elipsometryczne ($\Psi$ i $\Delta$) dla modelu:
    Ambient / Warstwa EMA (Szorstkość) / Warstwa Właściwa / Podłoże (Si).

    Zwraca:
        tuple: Wektory (psi_spectrum, delta_spectrum) w stopniach.
    """
    th_0 = np.radians(angle_deg)
    psi_spectrum = []
    delta_spectrum = []

    for i, wl in enumerate(wavelengths_nm):
        N_si = n_si[i] + 1j * k_si[i]
        N_mat = n_mat[i] + 1j * k_mat[i]
        N_ambient = 1.0 + 0j

        N_ema = bruggeman_ema(N_mat, N_ambient, f_material=0.5)

        n_list = [N_ambient, N_ema, N_mat, N_si]
        d_list = [np.inf, roughness_nm, thickness_nm, np.inf]

        res_s = coh_tmm('s', n_list, d_list, th_0, wl)
        res_p = coh_tmm('p', n_list, d_list, th_0, wl)

        rho = res_p['r'] / res_s['r']

        psi = np.arctan(np.abs(rho))
        delta = np.angle(rho)

        if delta < 0:
            delta += 2 * np.pi

        psi_spectrum.append(np.degrees(psi))
        delta_spectrum.append(np.degrees(delta))

    return np.array(psi_spectrum), np.array(delta_spectrum)


if __name__ == "__main__":
    # Test jednostkowy symulatora fizycznego
    fetcher = MaterialOpticalConstants()
    n_si, k_si = fetcher.get_nk('main', 'Si', 'Franta-25C')
    n_sio2, k_sio2 = fetcher.get_nk('main', 'SiO2', 'Malitson')

    thickness = 100.0
    roughness = 2.0
    angle = 70.0

    wavelengths = 1239.84 / fetcher.energy_grid

    psi, delta = calculate_ellipsometry(
        n_si, k_si, n_sio2, k_sio2, thickness, roughness, wavelengths, angle
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fetcher.energy_grid, psi, color='blue', linewidth=2)
    plt.title(rf'Widmo $\Psi$ - SiO2 {thickness}nm')
    plt.xlabel('Energia (eV)')
    plt.ylabel(r'$\Psi$ (stopnie)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fetcher.energy_grid, delta, color='red', linewidth=2)
    plt.title(rf'Widmo $\Delta$ - SiO2 {thickness}nm')
    plt.xlabel('Energia (eV)')
    plt.ylabel(r'$\Delta$ (stopnie)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()