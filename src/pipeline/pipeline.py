import click
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

@click.command()
@click.option("--output", "-o", default="model.joblib", help="Ruta de salida del modelo")
def cli(output: str):
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"Accuracy: {acc:.3f}")
    joblib.dump(pipe, output)
    print(f"Modelo guardado en: {output}")
