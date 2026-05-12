import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from src.dataframe_processing import creating_dataframe


def model_training() -> None:
    df = creating_dataframe()

    x = df.drop("is_drop", axis = 1)
    y = df["is_drop"]

    x_train, x_text, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)

    scaler = StandardScaler()
    x_train_scaler = scaler.fit_transform(x_train)
    x_test_scaler = scaler.transform(x_text)

    model = SVC(kernel="rbf", probability=True, class_weight='balanced', C = 10)
    model.fit(x_train_scaler, y_train)

    y_pred = model.predict(x_test_scaler)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'drop_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return None
