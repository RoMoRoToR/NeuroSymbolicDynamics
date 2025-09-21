import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import welch

def main(csv="data/karman_val.csv", fs=100.0, out="results/karman/physics.csv"):
    df = pd.read_csv(csv)
    CL, CD = df["CL"].values, df["CD"].values
    f, Pxx = welch(CL, fs=fs, nperseg=2048)
    f_peak = f[np.argmax(Pxx)]
    st = f_peak  # при D=U=1 безразмерно; иначе St = f_peak*D/U_inf
    cl_rms = CL.std()
    cd_mean = CD.mean()
    import os; os.makedirs("results/karman", exist_ok=True)
    pd.DataFrame({"St":[st], "CL_rms":[cl_rms], "CD_mean":[cd_mean]}).to_csv(out, index=False)
    plt.figure(); plt.semilogy(f, Pxx); plt.axvline(f_peak, ls="--"); plt.xlabel("f"); plt.ylabel("PSD(CL)")
    plt.title(f"St ≈ {st:.3f}"); plt.tight_layout(); plt.savefig("results/karman/PSD_CL.png", dpi=140)
    print(f"[ok] saved physics metrics to {out}")
if __name__ == "__main__": main()
