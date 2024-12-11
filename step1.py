#!/usr/bin/env python
# coding: utf-8

# <h1> Dataset

# In[21]:


import numpy as np
import pandas as pd

manuale = pd.read_csv('manuale.csv', sep=';')
manuale


# Consideriamo anche il dataset ripulito dai valori anomali

# In[ ]:


#PULIZIA DEL DATA SET DA RIVEDERE

# Calcolare la mediana dei valori non anomali
manualePulito = pd.read_csv('manuale.csv', sep=';')
bs_median = manualePulito.loc[manualePulito['BS'] != 701, 'BS'].median()

# Sostituzione dei valori anomali (701) con la mediana calcolata
manualePulito['BS'] = manualePulito['BS'].replace(701, bs_median)
manuale



# Modelli utilizzati:
# <span style="color:red"> KNN , Gaussian Naive Bayes</span> 

# <h1> K-Nearest-Neighbors

# La formula della distanza euclidea tra due punti in uno spazio n-dimensionale è:
# 
# $$
# d(p_1, p_2) = \sqrt{\sum_{i=1}^{n} (p_{1,i} - p_{2,i})^2}
# $$
# 
# Per uno spazio bidimensionale, la formula diventa:
# 
# $$
# d(p_1, p_2) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
# $$

# In[24]:


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):  # Escludere l'ultima colonna (la classe target)
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


# La funzione knn_predict implementa l’algoritmo KNN. Dato un set di punti di training  X , le rispettive etichette  y , un set di punti di test e  k , restituisce le predizioni.

# In[25]:


from collections import Counter

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Calcola la distanza tra il punto di test e tutti i punti di training
        distances = [(euclidean_distance(test_point, x_train), y_train[i]) for i, x_train in enumerate(X_train)]
        # Ordina le distanze in ordine crescente
        distances.sort(key=lambda x: x[0])
        # Seleziona i primi k vicini
        k_nearest = [dist[1] for dist in distances[:k]]
        # Predici la classe in base alla votazione a maggioranza
        most_common = Counter(k_nearest).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions


# Cross-validation con k-fold : divide il dataset in  k -fold, usa ciascun fold come test set una volta, mentre il resto dei fold viene usato come training set. Successivamente calcoliamo l’accuratezza media su tutti i fold.

# In[26]:


# Funzione per suddividere i dati in k-fold
def k_fold_split(X, y, k):
    fold_size = len(X) // k
    X_folds = []
    y_folds = []
    for i in range(k):
        X_folds.append(X[i * fold_size: (i + 1) * fold_size])
        y_folds.append(y[i * fold_size: (i + 1) * fold_size])
    return X_folds, y_folds

def cross_validate_knn(X, y, k_neighbors, num_folds=5):
    X_folds, y_folds = k_fold_split(X, y, num_folds)
    
    accuracies = []  # Per memorizzare le accuratezze per ciascun fold
    for i in range(num_folds):
        # Usa il fold i-esimo come test set
        X_test_fold = X_folds[i]
        y_test_fold = y_folds[i]
        
        # Usa tutti gli altri fold come training set
        X_train_folds = np.concatenate([X_folds[j] for j in range(num_folds) if j != i])
        y_train_folds = np.concatenate([y_folds[j] for j in range(num_folds) if j != i])
        
        # Prevedi con KNN
        y_pred = knn_predict(X_train_folds, y_train_folds, X_test_fold, k_neighbors)
        
        # Calcola l'accuratezza per questo fold
        accuracy = np.mean(np.array(y_pred) == np.array(y_test_fold))
        accuracies.append(accuracy)
    
    # Restituisci l'accuratezza media sui fold
    return np.mean(accuracies)


# Utilizziamo una funzione per trovare il miglior  k.  Testiamo diversi valori di  k  e valutiamo le prestazioni medie tramite cross-validation

# In[27]:


def find_best_k(X, y, k_values, num_folds=5):
    best_k = k_values[0]
    best_accuracy = 0
    
    for k in k_values:
        # Eseguiamo la cross-validation per ciascun k
        accuracy = cross_validate_knn(X, y, k, num_folds)
        print(f'Accuracy for k={k}: {accuracy:.4f}')
        
        # Se l'accuratezza per questo k è migliore, aggiorna il miglior k
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print(f'Miglior valore di k: {best_k} con un\'accuratezza di {best_accuracy:.4f}')
    return best_k


# In[46]:


# Separa le feature (X) e le etichette (y)
X = manuale.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima (feature)
y = manuale.iloc[:, -1].values    # L'ultima colonna (target)
# Esegui la ricerca del miglior valore di k
k_values = list(range(1, 11))  # Prova k da 1 a 10
best_k = find_best_k(X, y, k_values, num_folds=5)  # Cross-validation a 5 fold

# Valutazione finale: Uso del miglior k trovato per fare predizioni su tutto il dataset
y_final_pred = knn_predict(X, y, X, best_k)

# Calcola e stampa l'accuratezza finale usando lo stesso dataset
final_accuracy = np.mean(y_final_pred == y)
print(f'Accuratezza finale sui dati (con k={best_k}): {final_accuracy:.4f}')


# In[47]:


#proviamo con il dataset manualePulito 

# Separa le feature (X) e le etichette (y) però sul dataset ripulito 
Xp = manualePulito.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima (feature)
yp = manualePulito.iloc[:, -1].values    # L'ultima colonna (target)
# Esegui la ricerca del miglior valore di k
k_values = list(range(1, 11))  # Prova k da 1 a 10
best_k = find_best_k(Xp, yp, k_values, num_folds=5)  # Cross-validation a 5 fold

# Valutazione finale: Uso del miglior k trovato per fare predizioni su tutto il dataset
y_final_pred = knn_predict(Xp, yp, Xp, best_k)

# Calcola e stampa l'accuratezza finale usando lo stesso dataset
final_accuracy = np.mean(y_final_pred == yp)
print(f'Accuratezza finale sui dati (con k={best_k}): {final_accuracy:.4f}')


# <h1> Gaussian Naive Bayes

# In[13]:


def calcola_parametri(X, y):
    # Identifico le classi uniche
    classes = np.unique(y)
    
    # Dizionario per contenere i parametri per ogni classe
    parameters = {}
    
    # Calcola media, varianza e probabilità a priori per ogni classe
    for cls in classes:
        X_c = X[y == cls]  # Dati appartenenti alla classe corrente
        parameters[cls] = {
            'mean': X_c.mean(axis=0),  # Media di ciascuna feature
            'var': X_c.var(axis=0),    # Varianza di ciascuna feature
            'prior': X_c.shape[0] / X.shape[0]  # Probabilità a priori
        }
    
    return parameters


# In[14]:


def gaussian_probability(x, mean, var):
    # Aumenta epsilon per evitare varianze troppo piccole che causano problemi numerici
    epsilon = 1e-1 # Valore piccolo ma più grande di quello precedente
    coefficient = 1 / np.sqrt(2 * np.pi * (var + epsilon))
    exponent = np.exp(-((x - mean) ** 2) / (2 * (var + epsilon)))
    return coefficient * exponent


# In[15]:


def class_probability(x, parameters):
    probabilities = {}
    
    # Calcola la probabilità per ciascuna classe
    for cls, params in parameters.items():
        prior = np.log(params['prior'])  # Usa log per la probabilità a priori
        likelihood = np.sum(np.log(gaussian_probability(x, params['mean'], params['var'])))
        probabilities[cls] = prior + likelihood
    
    return probabilities


# In[16]:


def predict(X, parameters):
    predictions = []
    for x in X: # Per ogni esempio nei dati di input
        class_probabilities = class_probability(x, parameters)
        # Seleziona la classe con la probabilità a posteriori massima
        predictions.append(max(class_probabilities, key=class_probabilities.get))
    
    return np.array(predictions)


# In[ ]:


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)  # Conta le predizioni corrette
    accuracy = correct_predictions / len(y_true)  # Divide per il numero totale di esempi
    return accuracy


# In[30]:


# con il dataset non ripulito 
# Calcolo dei parametri (media, varianza, probabilità a priori) per ogni classe 
parameters = calcola_parametri(X, y)

X_test = X  # Utilizzo dati esistenti

# Previsione  delle classi
predictions = predict(X_test, parameters)

# Visualizza le predizioni
print("Predizioni:", predictions)

# Confronta con le etichette vere 
print("Etichette vere:", y)


# In[44]:


predictions = predict(X, parameters)  # Prevediamo con il modello Gaussian Naive Bayes

# Valutazione dell'accuratezza
accuracy = calculate_accuracy(y, predictions)
print(f"Accuracy: {accuracy:.2f}")


# In[48]:


parameters_p = calcola_parametri(Xp, yp)
predictions_p = predict(Xp, parameters_p)  # Prevediamo con il modello Gaussian Naive Bayes

# Valutazione dell'accuratezza
accuracy = calculate_accuracy(yp, predictions_p)
print(f"Accuracy: {accuracy:.2f}")


# Matrice di confusione

# In[51]:


def confusion_matrix(y_true, y_pred):
    # Identifica le classi uniche
    classes = np.unique(np.concatenate((y_true, y_pred)))
    # Crea una matrice di confusione vuota
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Popola la matrice
    for true, pred in zip(y_true, y_pred):
        true_index = np.where(classes == true)[0][0]  # Trova l'indice della classe vera
        pred_index = np.where(classes == pred)[0][0]  # Trova l'indice della classe prevista
        matrix[true_index, pred_index] += 1
    
    return matrix, classes

# Usa la funzione per calcolare la matrice di confusione
cm, classes = confusion_matrix(y, predictions)
cm_p, classes = confusion_matrix(yp, predictions_p)

# Stampa la matrice di confusione
print("Matrice di confusione:")
print(cm)

print(cm_p)

# Stampa le etichette delle classi
print("Etichette delle classi:", classes)

