

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):  # Escludere l'ultima colonna (la classe target)
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

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

def gaussian_probability(x, mean, var):
    # Aumenta epsilon per evitare varianze troppo piccole che causano problemi numerici
    epsilon = 1e-1 # Valore piccolo ma più grande di quello precedente
    coefficient = 1 / np.sqrt(2 * np.pi * (var + epsilon))
    exponent = np.exp(-((x - mean) ** 2) / (2 * (var + epsilon)))
    return coefficient * exponent

def class_probability(x, parameters):
    probabilities = {}
    
    # Calcola la probabilità per ciascuna classe
    for cls, params in parameters.items():
        prior = np.log(params['prior'])  # Usa log per la probabilità a priori
        likelihood = np.sum(np.log(gaussian_probability(x, params['mean'], params['var'])))
        probabilities[cls] = prior + likelihood
    
    return probabilities

def predict(X, parameters):
    predictions = []
    for x in X: # Per ogni esempio nei dati di input
        class_probabilities = class_probability(x, parameters)
        # Seleziona la classe con la probabilità a posteriori massima
        predictions.append(max(class_probabilities, key=class_probabilities.get))
    
    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)  # Conta le predizioni corrette
    accuracy = correct_predictions / len(y_true)  # Divide per il numero totale di esempi
    return accuracy

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


