import numpy as np
import matplotlib.pyplot as plt

f = 5.0e9           # Частота, Гц
ratio = 0.9         # Отношение 2l/λ
c = 3e8             # Скорость света, м/с

lmbd = c / f        # Длина волны, м
l = ratio * lmbd / 2 # Длина одного плеча, м
k = 2 * np.pi / lmbd # Волновое число

theta = np.linspace(1e-9, np.pi - 1e-9, 2000)

def E(theta):
    num = np.cos(k * l * np.cos(theta)) - np.cos(k * l)
    den = np.sin(theta)
    return num / den

def F(theta):
    return np.abs(E(theta)) / np.max(np.abs(E(theta)))

def Dmax(theta):
    formula = F(theta)**2 * np.sin(theta)
    if hasattr(np, 'trapezoid'):
        integral = np.trapezoid(formula, theta)
    else:
        integral = np.trapz(formula, theta)
    return 2.0 / integral  # 4π / (2π·∫) = 2 / ∫

def D(theta):
    return F(theta)**2 * Dmax(theta)

def creating_plot(d_times, d_dB, theta):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'polar': False})
    fig.suptitle( 'D(θ) - Вариант 5  (f=5.0 ГГц, 2l/λ=0.9)')

    axs[0, 0].plot(theta, d_times, color='blue', label='Аналитический расчёт')
    axs[0, 0].set_title("КНД (разы, декарт)")
    axs[0, 0].set_xlabel("θ, рад")
    axs[0, 0].set_ylabel("D(θ)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(theta, d_dB, color='red', label='Аналитический расчёт')
    axs[0, 1].set_title("КНД (дБ, декарт)")
    axs[0, 1].set_xlabel("θ, рад")
    axs[0, 1].set_ylabel("D(θ) [дБ]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0] = plt.subplot(2, 2, 3, polar=True)
    axs[1, 0].plot(theta, d_times, color='blue', label='Аналитический расчёт')
    axs[1, 0].set_title("КНД (разы, поляр)")
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='upper right')

    axs[1, 1] = plt.subplot(2, 2, 4, polar=True)
    d_dB_polar = np.clip(d_dB, -30, None)  # Ограничение для наглядности
    axs[1, 1].plot(theta, d_dB_polar, color='red', label='Аналитический расчёт')
    axs[1, 1].set_title("КНД (дБ, поляр)")
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('task2var5.png', dpi=300)
    plt.show()

def main():
    print(f'{Dmax(theta=theta):.3f} раз')
    print(f'{10*np.log10(Dmax(theta=theta)):.3f} дБ')
    print('-' * 30)

    d_times = D(theta)
    d_dB = 10 * np.log10(d_times + 1e-12)  # Защита от log(0)
    
    creating_plot(d_times=d_times, d_dB=d_dB, theta=theta)

    with open('analyse_results.txt', 'w', encoding='utf-8') as file:
        file.write('theta_rad  D_linear  D_dB\n')
        for i in range(len(theta)):
            file.write(f'{theta[i]} {d_times[i]} {d_dB[i]}\n')
            
    print("✓ Графики сохранены: task2var5.png")
    print("✓ Данные сохранены: analyse_results.txt")

if __name__ == "__main__":
    main()