inteligencia = 9
dano = 25

if inteligencia >= 7:
    dano += 5
else:
    dano -= 2

print(f"dano novo da arma: {dano}")

danos = [dano + 5 for dano in range(2) if inteligencia >= 7]

print(danos)