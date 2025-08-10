from src.data_loader import load_data
from src.models import train_linear_regression, train_random_forest
from src.metrics import evaluate_model


def display_results(name, metrics):
    mae, mse, r2 = metrics
    print(f"\n{name} Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")


def main():
    X_train, X_test, y_train, y_test = load_data("C:/Users/Aryan/OneDrive/Desktop/Projects/EnergiSense/CCPP/Folds5x2.csv")

    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    display_results("Linear Regression", lr_metrics)

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    display_results("Random Forest", rf_metrics)


if __name__ == "__main__":
    main()
