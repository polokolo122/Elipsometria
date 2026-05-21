# Deep Learning Ellipsometry 2.0

Rozszerzony projekt naukowo-badawczy dotyczący zastosowania probabilistycznego uczenia maszynowego (Bayesian Deep Learning) oraz sieci neuronowych ukierunkowanych fizycznie (Physics-Informed Neural Networks - PINN) do automatyzacji problemu odwrotnego w elipsometrii spektroskopowej.

Zbudowany system analizuje surowe dane pomiarowe ($\Psi$ i $\Delta$) w celu określenia rodzaju materiału badanej warstwy, jej grubości oraz szorstkości powierzchni. Najważniejszą innowacją w stosunku do wersji podstawowej jest zdolność sieci do szacowania **fizycznej niepewności pomiaru (przedziału ufności)**, co jest kluczowe w rygorystycznych analizach metrologicznych.

## 💡 Uzasadnienie Aktualizacji

Wersja 1.0 projektu skutecznie zademonstrowała, że sieci neuronowe (1D-CNN) mogą z dużą precyzją odtwarzać parametry warstw cienkich z widm optycznych. Z punktu widzenia rygoru naukowego i inżynierii pomiarowej posiadała jednak ograniczenia, które w nowej wersji zostały wyeliminowane:

1. **Problem Ślepego Zaufania (Brak oceny niepewności):** Klasyczne modele sztucznej inteligencji podają absolutne wartości, nawet jeśli informacja fizyczna jest niedostępna (np. brak powracającego sygnału z podłoża przy materiałach silnie absorbujących, jak TiN). W metrologii wynik pozbawiony wyliczonego błędu pomiarowego jest naukowo niekompletny.
2. **Brak Matematycznego Sprzężenia Zwrotnego:** Wersja podstawowa nie posiadała mechanizmu sprawdzającego, czy parametry zgadywane przez sieć faktycznie odpowiadają prawom fizyki.
3. **Standaryzacja Danych (Data Leakage):** Z perspektywy wdrażania modelu na produkcję (MLOps), proces skalowania danych w wersji 1.0 wymagał optymalizacji, aby zapobiec wyciekowi informacji ze zbioru testowego do zbioru uczącego.

## 🚀 Wprowadzone Innowacje (Metodologia)

Wersja 2.0 wprowadza szereg zaawansowanych mechanizmów obliczeniowych:

* **Ocena Niepewności (Uncertainty Quantification):** Głowica regresyjna w architekturze 1D-CNN została przebudowana. Obecnie estymuje nie tylko wartość oczekiwaną parametru ($\mu$), ale również jego wariancję ($\sigma^2$), co pozwala raportować wynik w formie "Wartość $\pm$ Błąd".
* **Funkcja Straty NLL (Gaussian Negative Log Likelihood):** Zastąpiono tradycyjne funkcje straty (MSE/MAE) podejściem probabilistycznym. Sieć optymalizuje teraz rozkład Gaussa, co zmusza model do kalibracji pewności swoich przewidywań proporcjonalnie do trudności sygnału wejściowego.
* **Walidacja PINN (Physics-Informed Validation):** Opracowano nowy moduł (`physics_validator.py`). Narzędzie to dekoduje parametry za pomocą sieci neuronowej, a następnie podaje je jako wejście do pierwotnego modelu fizycznego (Transfer Matrix Method). Pozwala to wyrysować i wizualnie porównać zrekonstruowane widmo teoretyczne z oryginalnym pomiarem, domykając cykl weryfikacyjny.
* **Architektura MLOps:** Wdrożono rygorystyczny proces zapisywania statystyk populacyjnych (skalerów danych) w formie obiektów `.pkl` podczas treningu. Gwarantuje to absolutną poprawność i wiarygodność wyników na całkowicie nowych (nieznanych modelowi) próbkach.

## 📂 Zawartość Projektu

Projekt dzieli się na logiczne moduły odpowiedzialne za zbieranie danych, symulację zjawisk fizycznych i trening modelu.

* `data_fetcher.py` – Interfejs automatycznie pobierający zinterpolowane stałe optyczne ($n$, $k$) z centralnej bazy danych *refractiveindex.info*.
* `dataset_generator.py` – Solwer fizyczny oparty na metodzie macierzy przejścia (TMM) uwzględniający przybliżenie efektywnego ośrodka Bruggemana (EMA) dla określenia wpływu chropowatości.
* `generate_dataset.py` – Moduł generacji syntetycznych zbiorów danych (Big Data) naśladujących rzeczywiste pomiary, z wbudowanymi ograniczeniami wynikającymi z fizycznej głębokości wnikania promieniowania świetlnego.
* `train_model.py` – Skrypt treningowy wdrażający procedury probabilistycznego uczenia głębokiego (sieć 1D-CNN generująca wagi dla log-wiarygodności) z jednoczesną serializacją modelu `.pth` oraz transformatorów `.pkl`.
* `test_model.py` – Izolowane środowisko ewaluacyjne oceniające globalny błąd wyznaczania geometrii układu (MAE), z wdrożonym raportowaniem odchylenia standardowego ($\sigma$) każdego pomiaru.
* `physics_validator.py` – **[New]** System sprzężenia zwrotnego. Porównuje na jednym wykresie surowe dane spektroskopowe ze zrekonstruowanym rozwiązaniem teoretycznym wyliczonym na podstawie predykcji modelu AI.

## 🛠 Instalacja i Wymagania Środowiskowe

Kod przetestowano w środowisku Python 3.9+. Należy zapewnić instalację poniższych zależności:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm tmm refractiveindex joblib
