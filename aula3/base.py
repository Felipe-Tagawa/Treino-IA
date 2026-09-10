import numpy as np

valores = np.array([[1,2,3,4,5],
                   [6,7,8,9,10]])

print(np.max(valores))
print(np.mean(valores))
print(np.argmax(valores))
print(np.shape(valores))

notas = np.array([50,67,88,91,100])

print(f"Média: {np.mean(notas)}")
print(f"Maior Nota: {np.max(notas)}")
print(f"Menor Nota: {np.min(notas)}")
print(f"Soma: {np.sum(notas)}")