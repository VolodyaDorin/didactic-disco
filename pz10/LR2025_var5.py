import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
import json
import yaml

class RCSConfig:
    """Читает параметры варианта из YAML-файла."""
    def __init__(self, filename: str, variant: int):
        self.filename = filename
        self.variant = variant
        self.D = None
        self.fmin = None
        self.fmax = None
        self.load()

    def load(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)          # data — это список словарей
        for item in data:
            if item['variant'] == self.variant:
                self.D = float(item['D'])
                self.fmin = float(item['fmin'])
                self.fmax = float(item['fmax'])
                return
        # Если ничего не нашли — ошибка
        raise ValueError(f"Вариант {self.variant} не найден в файле {self.filename}")

class RCSCalculator:
    """Вычисляет ЭПР идеально проводящей сферы по ряду Ми."""
    def __init__(self, radius: float, n_max: int = 50):
        self.r = radius
        self.n_max = n_max

    def _hankel(self, n, x):
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)

    def calculate_sigma(self, frequency: float) -> float:
        c = 3e8
        lam = c / frequency
        k = 2 * np.pi / lam
        kr = k * self.r

        s = 0 + 0j
        for n in range(1, self.n_max + 1):
            jn = spherical_jn(n, kr)
            jn_1 = spherical_jn(n - 1, kr)
            hn = self._hankel(n, kr)
            hn_1 = self._hankel(n - 1, kr)

            a_n = jn / hn
            b_n = (kr * jn_1 - n * jn) / (kr * hn_1 - n * hn)

            term = ((-1) ** n) * (n + 0.5) * (b_n - a_n)
            s += term

        sigma = (lam ** 2 / np.pi) * np.abs(s) ** 2
        return sigma.real

class ResultWriter:
    """Сохраняет результаты в JSON."""
    def __init__(self, filename: str):
        self.filename = filename

    def write(self, frequencies: np.ndarray, sigmas: np.ndarray):
        c = 3e8
        data = []
        for f, rcs in zip(frequencies, sigmas):
            data.append({
                "freq_Hz": f,
                "lambda_m": c / f,
                "rcs_m2": rcs
            })
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в {self.filename}")

def plot_rcs(frequencies, sigmas):
    """Рисует и сохраняет график ЭПР от частоты."""
    plt.figure(figsize=(9, 5))
    plt.plot(frequencies / 1e9, sigmas, linewidth=1.2)
    plt.xlabel("Частота, ГГц")
    plt.ylabel("ЭПР, м²")
    plt.title("Зависимость ЭПР идеально проводящей сферы от частоты")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("rcs_sphere_variant5.png", dpi=150)
    plt.show()

def main():
    # ----- МЕНЯЕТСЯ ТОЛЬКО НОМЕР ВАРИАНТА -----
    variant_number = 5
    # -----------------------------------------

    # Загружаем данные для нужного варианта
    config = RCSConfig("task_rcs_01.yaml", variant=variant_number)
    r = config.D / 2.0
    fmin = config.fmin
    fmax = config.fmax

    # Генерируем частоты (логарифмическая шкала)
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num=300)

    # Считаем ЭПР
    calculator = RCSCalculator(radius=r, n_max=50)
    sigmas = np.array([calculator.calculate_sigma(f) for f in frequencies])

    # Сохраняем результат в JSON
    writer = ResultWriter("rcs_results_variant5.json")
    writer.write(frequencies, sigmas)

    # Строим график
    plot_rcs(frequencies, sigmas)

if __name__ == "__main__":
    main()