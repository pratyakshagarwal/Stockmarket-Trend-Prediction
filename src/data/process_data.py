import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------- Log Transformation -----------------------------------------------------
class LogTransformation:
    def apply_transformation(self, dataframe:pd.DataFrame, features:List) -> pd.DataFrame:
        """
        transform dataframe using LogTransformation 
        
        Parameters:
        dataframe (pd.DataFrame): The dataframe which contains data
        features (List): The list of feature which we have to scale

        Return:
        pd.DataFrame: The scaled dataframe
        """
        df_scaled = dataframe.copy()
        for feature in features:
            df_scaled[feature] = np.log1p(df_scaled[feature])
        logging.info(f"Applied log transformation to {features}")
        return df_scaled
    
    def reverse_transformation(self, dataframe:pd.DataFrame, features:List) -> pd.DataFrame:
        df_scaled = dataframe.copy()
        for feature in features:
            df_scaled[feature] = np.exp(df_scaled[feature] - 1) 
        logging.info(f"Applied Exp transformation to {features}")
        return df_scaled

# ----------------------------------------- make target column ---------------------------------------------------------------------
def create_target_columns(df:pd.DataFrame) -> pd.DataFrame:
    df['trend'] = (df['Close'] < df['Close'].shift(-1)).astype(int)
    return df

# ------------------------------------ Train scalers ---------------------------------------------------------------------------
def fit_scalers(dataframe: pd.DataFrame, stnd_scaler_columns: List[str]) -> Dict[str, BaseEstimator]:
    scalers = {}
    for column in stnd_scaler_columns:
        if column in dataframe.columns:
            scaler = StandardScaler()
            scaler.fit(dataframe[column].astype(float).values.reshape(-1, 1))
            scalers[column] = scaler
        else:
            logging.info(f"Column '{column}' not found in dataframe.")
    return scalers


# ---------------------------------- Tranform data ------------------------------------------------------------------------------
def scale_data(dataframe: pd.DataFrame, scalers: Dict[str, BaseEstimator]) -> pd.DataFrame:
    dataframe_copy = dataframe.copy()
    for column in dataframe_copy.columns:
        if column in scalers:
            try:
                dataframe_copy[column] = scalers[column].transform(dataframe_copy[column].astype(float).values.reshape(-1, 1))
            except Exception as e:
                logging.error(f"Error scaling column '{column}': {e}")
                # Optionally, handle the error or set the column to NaN/zero, etc.
                dataframe_copy[column] = None
        else:
            logging.info(f"No scaler found for column '{column}'")
    return dataframe_copy


# ---------------------------------- Arrange data in specfic column order --------------------------------------------------------
def arrange_dataframe(dataframe:pd.DataFrame, columns_order:List[str]) -> pd.DataFrame:
    logging.info(f"Columns arranged in format:{columns_order}")
    return dataframe[columns_order]


# ---------------------------------- make data model feedable --------------------------------------------------------------------
def make_data(dataframe: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    
    # Convert dataframe to numpy array for efficiency
    data = dataframe.values

    # Loop to create sequences
    for i in range(len(data) - window_size):
        features.append(data[i:i + window_size, :-1])  # All columns except the last (features)
        targets.append(data[i + window_size, -1])  # Last column (target)
    
    # Convert lists to numpy arrays for model compatibility
    return np.array(features), np.array(targets)

# --------------------------------- split data into train, test and forecast -----------------------------------------------------------
def split_data(dataframe:pd.DataFrame, train_size:int=0.80, test_size:int=0.15) -> List[pd.DataFrame]:
    tz = int(len(dataframe) * train_size)
    vz = int(len(dataframe) * test_size)

    logging.info(f"Splitting data with train size: {train_size} test size {test_size}")

    train_data = dataframe.iloc[:tz, :]
    validation_data = dataframe.iloc[tz:tz+vz, :]
    forecast_data = dataframe.iloc[tz+vz:, :]

    return train_data, validation_data, forecast_data


# ---------------------------------- save and load scalers and dataframe ---------------------------------------------------------------- 
def save_scalers(scalers:Dict[str, BaseEstimator], path) -> None:
    joblib.dump(scalers, path)
    logging.info(f"scalers saved to {path}")

def load_scalers(path:str) -> Dict[str, BaseEstimator]:
    scalers = joblib.load(path)
    logging.info(f"scalers loaded from {path}")
    return scalers

def save_dataframe(dataframe:pd.DataFrame, path:str) -> None:
    dataframe.to_csv(path, index=False)
    logging.info(f"dataframe saved to {path}")

if __name__ == '__main__':
    path = os.getcwd(); datafolder_path = os.path.join(path, "data")
    datafile_path = os.path.join(datafolder_path, os.listdir(datafolder_path)[0])

    df = pd.read_csv(datafile_path)
    df = create_target_columns(df=df)
    columns_order = pd.Index(['Open', 'High', 'Low', 'Volume', 'Close', 'trend'], dtype='object')
    arranged_df = arrange_dataframe(df, columns_order=columns_order)
    print(arranged_df.head())
    
    print(f"\n Describing data before normalizing:")
    print(arranged_df.describe())
    train_data, test_data, forecast_data = split_data(arranged_df, train_size=0.85, test_size=0.14)
    
    # scalers = fit_scalers(train_data, train_data.columns[:-1])
    # save_scalers(scalers, "models/stnd_scalers.pkl")

    print(f"\n Describiing train data befroe normalization:")
    print(train_data.describe())
    print(f"\n Describiing test data before normalization:")
    print(test_data.describe())

    log_transformer = LogTransformation()
    scaled_train_data = log_transformer.apply_transformation(train_data, features=['Open', 'High', 'Low', 'Volume', 'Close'])
    scaled_test_data = log_transformer.apply_transformation(test_data, features=['Open', 'High', 'Low', 'Volume', 'Close'])
    

    # scaled_train_data = scale_data(train_data, scalers)
    # scaled_test_data = scale_data(test_data, scalers)

    print(f"\n Describiing train data after normalization:")
    print(scaled_train_data.describe())
    print(f"\n Describiing test data after normalization:")
    print(scaled_test_data.describe())

    print(f"train data shape: {train_data.shape}\
          test data shape: {test_data.shape}\
            forecast data shape :{forecast_data.shape}")