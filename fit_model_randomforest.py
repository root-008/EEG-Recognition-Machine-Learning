from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import createX_y
import pickle

def fit_randomForest_with_kfold():
    X, y = createX_y.X_y_for_classif()
    kf = KFold(n_splits=10)

    best_model = None
    best_accuracy = 0
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        accuracies.append(acc)
        
        # En iyi modeli seç
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    # En iyi modeli dosyaya kaydet
    with open('savedModels/model_rf_kfold.pkl', 'wb') as model_file:
        pickle.dump((best_model, accuracies), model_file)
    

def fit_randomForest_with_holdout():
    X_train,_,X_test,y_train,_,y_test = createX_y.create_train_test_data_for_classif()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    model.fit(X_train,y_train)
    with open('savedModels/model_rf_holdout.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
