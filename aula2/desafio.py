nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
notas = [8, 6, 9, 5, 7, 10]
num_ap = 0
num_rep = 0
i = 0

alunos = dict(zip(nomes, notas))

for nome, nota in alunos.items():
    print(f"Nome: {nome}, Nota:{nota}")

while i < len(alunos):

    if notas[i] < 7:
        situacao = "Reprovado!"
        num_rep += 1
    else:
        situacao = "Aprovado!"
        num_ap += 1
    
    print(f"{nomes[i]}: Nota {notas[i]} - {situacao}\n")

    i += 1

print(f"Total de alunos: {len(alunos)}\n")
print(f"Número de Aprovados: {num_ap}\n")
print(f"Número de Reprovados: {num_rep}\n")
print(f"Aproveitamento do curso: {100 * (num_ap / len(alunos)):.2f} %\n")

