import pandas as pd
import numpy as np

alunos = {
    "Nome": ["Felipe", "Luigi", "Maria", "Pedro"],
    "Idade": [21, 22, 21, 20],
    "Nota": [6, 10, 9, 8],
    "Curso": ["GEC", "GET", "GEL", "GES"]
}

df = pd.DataFrame(alunos)

print(df.head(5))
print(df.tail(5))
print(f"Tamanho do DataFrame: {len(df)}")
print(df.columns)
print(df.info())
print(df.describe())
print(df["Nota"])
print(df[["Nome", "Nota"]])
print(df[df["Nota"]>=7])
print(df.sort_values("Nota"))
print(df.sort_values("Nota", ascending=False))
print(df.isnull().sum())