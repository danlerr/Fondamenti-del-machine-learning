from sklearn.ensemble import RandomForestClassifier

# Definisci il modello Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Parametri da ottimizzare (grid search)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Creazione del GridSearchCV per la ricerca dei migliori parametri
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# Allenamento tramite GridSearch
grid_search_rf.fit(X_train, y_train)

# Migliori parametri
print(f"Migliori parametri Random Forest: {grid_search_rf.best_params_}")

# Predizioni sul test set con i migliori parametri
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Calcolo delle metriche
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Accuratezza: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")




# Calcolare la validazione incrociata
cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring='accuracy')
print(f"Accuratezza media con cross-validation: {cv_scores.mean():.4f}")






importances = best_rf_model.feature_importances_
feature_names = X.columns  # Se X è un DataFrame
sorted_idx = importances.argsort()

# Grafico dell'importanza delle caratteristiche
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Importanza")
plt.title("Importanza delle caratteristiche nel modello Random Forest")
plt.show()