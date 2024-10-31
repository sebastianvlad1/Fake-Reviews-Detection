from data_preprocessing import load_and_prepare_data, split_data
from model_training import prepare_datasets, train_model
from evaluation import evaluate_model


def main():
    # Load and preprocess data
    data = load_and_prepare_data()
    X_train, X_val, y_train, y_val = split_data(data)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(X_train, X_val, y_train, y_val)

    # Train the model
    trainer = train_model(train_dataset, val_dataset)

    # Evaluate the model
    evaluate_model(trainer, val_dataset, y_val)


if __name__ == "__main__":
    main()
