import pandas as pd

notas = pd.Series([8,7,9,6])

print(notas)

alunos = {
    "Nome": ["Felipe", "Maria"],
    "Nota": [70, 100],
}

df = pd.DataFrame(alunos)

print(df["Nome"])
print(df[df["Nota"] > 70])
print(df.sort_values("Nota", ascending=True))
print(df.isnull().sum())
