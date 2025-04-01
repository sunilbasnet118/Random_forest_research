import src.data_preprocess as data_preprocess
import src.train as train
import src.actual_score as actual_score
import src.result as result

def main():
    # Data preprocessing
    X_train, y_train = data_preprocess.preprocess_data('dataset.json')

    # Train model
    model = train.train_model(X_train, y_train)

    # Save trained model
    train.save_model(model, 'trained_model.pkl')

    # Calculate actual score
    actual_score.calculate_actual_score(model, 'actual_dataset.json')

    # Visualize results
    result.main()

if __name__ == "__main__":
    main()