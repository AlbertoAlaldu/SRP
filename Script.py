import random
import statistics
import numpy as np
import matplotlib.pyplot as plt

def simulate_lifetime(
    rho,
    n_seeds=5000,
    T_max=400,
    gamma_ref=1.0,
    gamma_min=0.2,
    k0=0.25,
    alpha=0.9,
    mu=0.01,
    sigma0=0.18,
    c=0.06,
    U0=0.4,
):
    """Simula el tiempo de vida medio para un valor de rho."""
    lifetimes = []

    for _ in range(n_seeds):
        gamma = 1.0  # estado inicial
        for t in range(T_max):
            # Ganancia y saturación efectivas (se recortan con rho)
            k_eff = k0 * (1 - alpha * rho)
            U_max = U0 * (1 - alpha * rho)

            # Control tipo APP-lite
            u_raw = k_eff * (gamma_ref - gamma)
            u = max(min(u_raw, U_max), -U_max)

            # Entorno reducido
            eta = random.gauss(0, sigma0)
            e_t = mu + eta
            e_red = (1 - rho) * e_t

            # Dinámica de energía
            gamma = gamma + u + e_red - c

            # Condición de muerte
            if gamma < gamma_min:
                lifetimes.append(t + 1)
                break
        else:
            # Si no murió en T_max pasos, contamos T_max
            lifetimes.append(T_max)

    return statistics.mean(lifetimes)


# Barrido de rho
rhos = np.linspace(0.0, 1.0, 21)
W_vals = [simulate_lifetime(rho) for rho in rhos]

# Mostrar numeritos (opcional)
for r, w in zip(rhos, W_vals):
    print(f"rho = {r:.2f}, W(rho) ≈ {w:.1f} pasos")

# Graficar
plt.figure()
plt.plot(rhos, W_vals, marker="o")
plt.xlabel("rho (grado de reducción sistémica)")
plt.ylabel("W(rho) = tiempo de vida medio")
plt.title("Curva de viabilidad vs. reducción sistémica (PRS)")
plt.grid(True)
plt.show()
