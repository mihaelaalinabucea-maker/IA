from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data #atributele, lungimea petalei/separei
y = iris.target #clasele din care fac parte 0,1,2

print("Forma setului de date:", x.shape)

print("\nDenumirile atributelor:")
print(iris.feature_names)

print("\nClasele:")
print(iris.target_names)

# 2.1. Utilizați train_test_split() pentru a împărți datele: 80% antrenare, 20% testare.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 2.2. Afișați forma (shape) pentru fiecare subset.
print(f"\nForma subsetului de antrenare (x_train): {x_train.shape}")
print(f"Forma subsetului de testare (x_test): {x_test.shape}")
print(f"Forma etichetelor de antrenare (y_train): {y_train.shape}")
print(f"Forma etichetelor de testare (y_test): {y_test.shape}")

print(x)
print(y)
from sklearn.preprocessing import StandardScaler

# --- 3.1. Standardizarea caracteristicilor ---
# Inițializăm obiectul StandardScaler
scaler = StandardScaler()

# "Antrenăm" scaler-ul pe datele de antrenament și le transformăm
# Este important să folosim fit_transform pe train și doar transform pe test
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- 3.2. Compararea primelor 3 exemple înainte și după scalare ---
print("\n" + "="*30)
print("COMPARAȚIE DATE (Primele 3 rânduri)")
print("="*30)

print("\nÎnainte de scalare (x_train):")
print(x_train[:3])

print("\nDupă scalare (x_train_scaled):")
print(x_train_scaled[:3])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 4.1. Inițializarea modelului ---
# k=3 înseamnă că ne uităm la cei mai apropiați 3 vecini pentru a decide clasa
knn = KNeighborsClassifier(n_neighbors=3)

# --- 4.2. Antrenarea și Testarea ---
# Învățăm modelul folosind datele scalate (cele "echilibrate")
knn.fit(x_train_scaled, y_train)

# Punem modelul să prezică etichetele pentru datele de test pe care nu le-a văzut
y_pred = knn.predict(x_test_scaled)

# Calculăm acuratețea (cât la sută a ghicit corect)
accuracy = accuracy_score(y_test, y_pred)

print(f"Acuratețea modelului KNN pe setul de testare este: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

# Liste pentru a salva valorile k și acuratețea corespunzătoare
k_values = range(1, 16)
accuracies = []

# 5.1. Antrenarea și evaluarea pentru k între 1 și 15
for k in k_values:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(x_train_scaled, y_train)

    score = model_knn.score(x_test_scaled, y_test)
    accuracies.append(score)

# 5.2. Afișarea graficului
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
plt.title('Influența valorii K asupra Acurateții')
plt.xlabel('Valoarea lui K')
plt.ylabel('Acuratețe')
plt.xticks(k_values)
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 6.1. Afișarea matricei de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicție (Ce a crezut modelul)')
plt.ylabel('Realitate (Ce era de fapt)')
plt.title('Matricea de Confuzie')
plt.show()

# 6.2. Generarea raportului de clasificare
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nRaport de Clasificare:")
print(report)