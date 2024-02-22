from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score,roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

def metrics(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    
    # Choose an appropriate average setting for multiclass classification
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)

    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)

    disp_cm = ConfusionMatrixDisplay(conf_matrix)
    disp_cm.plot()

    plt.figure(figsize=(8, 4))
    bars = plt.bar(['Accuracy', 'F1 Score', 'Precision', 'Recall'], [accuracy, f1, precision, recall],
                   color=['blue', 'red', 'yellow', 'black'])
    
    plt.ylabel('Value')
    plt.ylim(0, 1)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 0.02, str(bar.get_height()), color='black')
    
    plt.show()

def ann_metrics(y_test, y_pred_probs):
    save_folder = 'gorseller/ann/'
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # One-hot encode y_test if it's not already
    if y_test.ndim == 2 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    unique_actual_classes = np.unique(y_test)
    unique_predicted_classes = np.unique(y_pred)
    print("Actual Unique Classes:", unique_actual_classes)
    print("Predicted Unique Classes:", unique_predicted_classes)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    disp_cm = ConfusionMatrixDisplay(conf_matrix)
    disp_cm.plot()
    plt.savefig(os.path.join(save_folder, 'cm.png'))

    plt.figure(figsize=(8, 4))
    bars = plt.bar(['Accuracy', 'F1 Score', 'Precision', 'Recall'], [accuracy, f1, precision, recall],
                   color=['blue', 'red', 'yellow', 'black'])
    
    plt.ylabel('Value')
    plt.ylim(0, 1)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 0.02, str(bar.get_height()), color='black')

    plt.savefig(os.path.join(save_folder, 'scores.png'))
    
    # ROC Curve
    # ROC Curve for each class
    plt.figure(figsize=(8, 4))
    n_classes = len(np.unique(y_test))
   
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
       
        plt.plot(fpr, tpr, lw=2, label='Class {} (ROC = {:.2f})'.format(i, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_folder, 'roc.png'))

   
    plt.show()


def loss_and_accuracy_graphic(history):
    save_folder = 'gorseller/ann/'
    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'loss_acc.png'))
    plt.show()


def kfold_plot(accuracies):
    plt.plot(accuracies, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy for K-Fold')
    plt.show()