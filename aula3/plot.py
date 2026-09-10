import matplotlib.pyplot as plt

Alunos = ["Evelyn", "Marcelo", "Pedro", "Gustavo", "Mateus", "Felipe"]
Periodo = [3, 4, 5, 6, 7, 8]


plt.bar(Alunos, Periodo)
plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Alunos x Períodos")
plt.show()
plt.hist(Periodo)
plt.xlabel("Período")
plt.title("Histograma dos Períodos dos Alunos")
plt.show()

Quantidade_Compradas = [1, 5, 10, 3, 8]
Itens = ["Feijão", "Arroz", "Batata", "Canela", "Palmito"]
plt.scatter(Itens, Quantidade_Compradas)
plt.xlabel("Produtos")
plt.ylabel("Quantidade de Produtos Vendidos")
plt.title("Relação de Dispersão entre quantidade e vendas")
plt.grid()
plt.show()