import numpy as np, pandas as pd

def synth_karman(T=20000, fs=100.0, f_strouhal=0.2, seed=0):
    """
    Генерирует ряды CL(t), CD(t), dP(t) с периодическим вихреотделением
    и редкими 'gusts' (интермиттентность), плюс несколько 'датчиков' в следе.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)/fs

    # базовый синус для shedding + амплитудная модуляция
    A = 0.8 + 0.2*np.sin(2*np.pi*0.01*t + 1.0)
    phi = 0.3
    CL = A * np.sin(2*np.pi*f_strouhal*t) + 0.05*rng.standard_normal(T)

    # CD имеет компоненту на 2*f_s (классика для цилиндра)
    CD = 1.0 + 0.15*np.sin(2*np.pi*2*f_strouhal*t + phi) + 0.05*rng.standard_normal(T)

    # перепад давлений как смесь гармоник
    dP = 0.6*np.sin(2*np.pi*f_strouhal*t - 0.5) + 0.25*np.sin(2*np.pi*3*f_strouhal*t + 0.2)
    dP += 0.05*rng.standard_normal(T)

    # редкие 'госты' (локальные импульсы амплитуды/частоты)
    n_gust = int(T/2000)+1
    for _ in range(n_gust):
        i0 = rng.integers(0, T-300)
        w = np.hanning(300)
        CL[i0:i0+300] += 0.6*w*rng.choice([-1,1])
        dP[i0:i0+300] += 0.3*w*rng.choice([-1,1])

    # несколько "сенсоров" в следе как фазовые/шумовые версии CL
    S1 = 0.7*A*np.sin(2*np.pi*f_strouhal*t + 0.6) + 0.08*rng.standard_normal(T)
    S2 = 0.5*A*np.sin(2*np.pi*f_strouhal*t - 0.9) + 0.08*rng.standard_normal(T)

    df = pd.DataFrame({
        "time": t, "CL": CL, "CD": CD, "dP": dP, "S1": S1, "S2": S2
    })
    return df

def split_save(df, out_prefix="data/karman"):
    import os; os.makedirs("data", exist_ok=True)
    n = len(df); n_train = int(0.6*n); n_val = int(0.2*n)
    df.iloc[:n_train].to_csv(f"{out_prefix}_train.csv", index=False)
    df.iloc[n_train:n_train+n_val].to_csv(f"{out_prefix}_val.csv", index=False)
    df.iloc[n_train+n_val:].to_csv(f"{out_prefix}_test.csv", index=False)
    print("[ok] saved train/val/test CSV to data/")

if __name__ == "__main__":
    df = synth_karman(T=30000, fs=100.0, f_strouhal=0.2, seed=42)
    split_save(df)
