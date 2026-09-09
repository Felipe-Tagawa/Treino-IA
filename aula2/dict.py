sabores_pizza = {
    "Chocolate": "doce",
    "Morango": "doce",
    "Margarita": "salgado",
    "Portuguesa": "salgado"
}

print(tuple(sabores_pizza.keys()))
print(list(sabores_pizza.values()))
print(sabores_pizza["Chocolate"][1]) # Imprimindo o 'O'

for i in range(len(sabores_pizza)):
    print(f"Chave {sabores_pizza.keys()}:{sabores_pizza.values}")