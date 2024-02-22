import pickle
import metrics
import createX_y
from tensorflow.keras.models import load_model
import pandas as pd

def load_knn(X_test,y_test,kfold):
    if kfold:
        with open('savedModels/model_knn_kfold.pkl', 'rb') as model_file:
            model, accuracies = pickle.load(model_file)
        
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred) 
        metrics.kfold_plot(accuracies)
    else:
        with open('savedModels/model_knn_holdout.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred)

def load_dt(X_test,y_test,kfold):
    if kfold:
        with open('savedModels/model_dt_kfold.pkl', 'rb') as model_file:
            model, accuracies = pickle.load(model_file)
        
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred) 
        metrics.kfold_plot(accuracies)
    else:
        with open('savedModels/model_dt_holdout.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred) 
        
def load_rf(X_test,y_test,kfold):
    if kfold:
        with open('savedModels/model_rf_kfold.pkl', 'rb') as model_file:
            model, accuracies = pickle.load(model_file)
        
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred) 
        metrics.kfold_plot(accuracies)
    else:
        with open('savedModels/model_rf_holdout.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(X_test)
        metrics.metrics(y_test, y_pred)
        
def load_ann(X_test, y_test):
    # Load history 
    history_df = pd.read_csv('savedModels/training_history.csv')
    metrics.loss_and_accuracy_graphic(history_df)

    model = load_model('savedModels/model_ann_holdout.keras')
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    metrics.ann_metrics(y_test, y_pred)

X_train,X_val,X_test,y_train,y_val,y_test = createX_y.create_train_test_data_for_ann()

load_ann(X_test, y_test)


