idade = 10

while idade < 18:
    print("Você é menor de idade!")
    idade += 1
    if idade <= 18:
        print("Feliz aniversário!")
    else:
        print("Você completou 18 anos!")

while(comando := input("Digite Algo (ou 'sair'):") != 'sair'):
    print(f"Você digitou {comando}")