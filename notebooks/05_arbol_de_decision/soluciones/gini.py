import pandas as pd
df = pd.DataFrame(
    {
        # S: Soleado, N: Nublado, L: Lluvioso
        "tiempo": ["S", "S", "N", "L", "L", "L", "N", "S", "S", "L", "S", "N", "N", "L"],
        # A: Alta, M: Media, B: Baja
        "temperatura": ["A", "A", "A", "M", "B", "B", "B", "M", "B", "M", "M", "M", "A", "M"],
        # A: Alta, N: Normal
        "humedad": ["A", "A", "A", "A", "N", "N", "N", "A", "N", "N", "N", "A", "N", "A"],
        # N: No, S: Sí
        "viento": ["N", "S", "N", "N", "N", "S", "S", "N", "N", "N", "S", "S", "N", "S"],
        # N: Negativo, P: Positivo
        "clase": ["N", "N", "P", "P", "P", "N", "P", "N", "P", "P", "P", "P", "P", "N"]
    }
)

def calculate_gi(df, col):
    casos = df.loc[:, col].unique()
    gi = 0
    for caso in casos:
        idx = df.loc[:, col] == caso
        peso = idx.sum() / len(df)
        p = (df.loc[idx, "clase"] == "P").sum() / idx.sum() 
        q = (df.loc[idx, "clase"] == "N").sum() / idx.sum()
        gi += peso * (1 - (p**2 + q**2))
    return gi

for col in df.columns:
    print(f"{col}: {calculate_gi(df, col):5.3f}")
