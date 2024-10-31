from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score


def evaluate_model(trainer, val_dataset, y_val):
    predictions, labels, _ = trainer.predict(val_dataset)
    y_pred = predictions.argmax(axis=1)

    print("Validation Results:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, predictions[:, 1]):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    return y_pred
