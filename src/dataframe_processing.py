import pandas as pd


from src.extract_features import song_processing


def creating_dataframe() -> pd.DataFrame:
    dataset_x, dataset_y = song_processing()

    columns = ['mean_rms', 'std_rms', 'mean_cent', 'std_cent', 'mean_zcr']
    df = pd.DataFrame(dataset_x, columns=columns)
    df["is_drop"] = dataset_y

    return df
