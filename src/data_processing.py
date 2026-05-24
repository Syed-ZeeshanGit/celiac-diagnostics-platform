# src/data_processing.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

class MedicalDataPipeline:
    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.mean_vectors: Any[np.ndarray] = None
        self.std_vectors: Any[np.ndarray] = None
        self.feature_columns: List[str] = []

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads clinical records, handles data sanitisation, executes one-hot encoding,
        and splits the dataset into training and testing evaluation matrices.
        """
        df = pd.read_csv(self.filepath)
        
        # Address missing elements safely in categorical streams
        df['Diabetes Type'] = df['Diabetes Type'].fillna('None')
        
        # Enforce deterministic binary mapping schemas
        binary_maps = {
            'Gender': {'Male': 0, 'Female': 1},
            'Diabetes': {'no': 0, 'Yes': 1},
            'Abdominal': {'no': 0, 'yes': 1},
            'Sticky_Stool': {'no': 0, 'yes': 1},
            'Weight_loss': {'no': 0, 'yes': 1},
            'Disease_Diagnose': {'no': 0, 'yes': 1}
        }
        for col, mapping in binary_maps.items():
            df[col] = df[col].map(mapping)
            
        # Execute structural one-hot encoding for nominal categories
        df = pd.get_dummies(df, columns=['Diabetes Type', 'Diarrhoea', 'Short_Stature'], drop_first=True)
        
        # Explicitly remove target labels and diagnostic descriptors to avoid data leakage
        self.feature_columns = [c for c in df.columns if c not in ['Marsh', 'cd_type', 'Disease_Diagnose']]
        
        X = df[self.feature_columns].astype(float).values
        y = df['Disease_Diagnose'].values
        
        # Isolate continuous variables for standardisation (Age, IgA, IgG, IgM)
        cont_cols = ['Age', 'IgA', 'IgG', 'IgM']
        cont_indices = [self.feature_columns.index(c) for c in cont_cols]
        
        # Compute and preserve training population scaling characteristics
        self.mean_vectors = X.mean(axis=0)
        self.std_vectors = X.std(axis=0)
        self.std_vectors[self.std_vectors == 0] = 1.0  # Safe boundary protection
        
        for idx in cont_indices:
            X[:, idx] = (X[:, idx] - self.mean_vectors[idx]) / self.std_vectors[idx]
            
        # Segment arrays into a structured 80/20 verification split
        np.random.seed(42)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split = int(0.8 * X.shape[0])
        
        return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

    def transform_single_input(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a live user interface input vector into a structured matrix 
        perfectly matching the population feature schema alignment.
        """
        df_single = pd.DataFrame([raw_input])
        
        # Structural conversion following equivalent construction rules
        df_single['Gender'] = df_single['Gender'].map({'Male': 0, 'Female': 1})
        df_single['Diabetes'] = df_single['Diabetes'].map({'no': 0, 'Yes': 1})
        df_single['Abdominal'] = df_single['Abdominal'].map({'no': 0, 'yes': 1})
        df_single['Sticky_Stool'] = df_single['Sticky_Stool'].map({'no': 0, 'yes': 1})
        df_single['Weight_loss'] = df_single['Weight_loss'].map({'no': 0, 'yes': 1})
        
        # Initialize one-hot sparse configurations explicitly
        nominal_options = {
            'Diabetes Type_Type 1': 1.0 if raw_input['Diabetes Type'] == 'Type 1' else 0.0,
            'Diabetes Type_Type 2': 1.0 if raw_input['Diabetes Type'] == 'Type 2' else 0.0,
            'Diarrhoea_inflammatory': 1.0 if raw_input['Diarrhoea'] == 'inflammatory' else 0.0,
            'Diarrhoea_watery': 1.0 if raw_input['Diarrhoea'] == 'watery' else 0.0,
            'Short_Stature_PSS': 1.0 if raw_input['Short_Stature'] == 'PSS' else 0.0,
            'Short_Stature_Variant': 1.0 if raw_input['Short_Stature'] == 'Variant' else 0.0,
        }
        
        for col, val in nominal_options.items():
            df_single[col] = val
            
        # Re-index array vectors to synchronize column sequence maps
        vec = np.zeros(len(self.feature_columns))
        for i, col in enumerate(self.feature_columns):
            if col in df_single.columns:
                vec[i] = float(df_single[col].iloc[0])
                
        # Scale continuous parameters using population parameters
        cont_cols = ['Age', 'IgA', 'IgG', 'IgM']
        for c in cont_cols:
            idx = self.feature_columns.index(c)
            vec[idx] = (vec[idx] - self.mean_vectors[idx]) / self.std_vectors[idx]
            
        return vec.reshape(1, -1)