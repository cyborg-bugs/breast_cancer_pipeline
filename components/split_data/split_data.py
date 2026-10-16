
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def split_data(encoded_scaled_data_path, X_train_path, X_test_path, y_train_path, y_test_path):
    print(f"Loading encoded and scaled data from: {encoded_scaled_data_path}")
    try:
        df_scaled = pd.read_csv(encoded_scaled_data_path)
        print("Encoded and scaled data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Encoded and scaled data not found at {encoded_scaled_data_path}")
        return
    except Exception as e:
        print(f"Error loading encoded and scaled data: {e}")
        return

    # Usar la columna objetivo directamente como la tienes tras el encode_scale
    target_col = 'Relapse Free Status'
    if target_col not in df_scaled.columns:
        raise KeyError(f"La columna objetivo '{target_col}' no se encuentra en el DataFrame. Columnas disponibles: {df_scaled.columns.tolist()}")

    # Binarizar la columna objetivo
    df_scaled[target_col] = df_scaled[target_col].map({'Recurred': 1, 'Not Recurred': 0})

    # Eliminar filas donde el target sea NaN después del mapeo
    initial_len = len(df_scaled)
    df_scaled = df_scaled.dropna(subset=[target_col])
    if len(df_scaled) < initial_len:
        print(f"Se eliminaron {initial_len - len(df_scaled)} filas que tenían target nulo tras el mapeo y se encontraban en el archivo encoded_scaled_data.csv.")

    # Separar features y target
    X = df_scaled.drop(columns=[target_col], errors='ignore')
    y = df_scaled[target_col]

    print("Features (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)
    print("Target distribution:\n", y.value_counts())

    # Split
    test_size = 0.2
    random_state = 42
    print(f"Splitting data into training (80%) and testing (20%) sets with random_state={random_state} and stratification...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("Data split successfully.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("Training target distribution:\n", y_train.value_counts())
        print("Testing target distribution:\n", y_test.value_counts())
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    # Guardar archivos resultantes en las rutas correspondientes
    output_paths = {
        'X_train': (X_train_path, X_train),
        'X_test': (X_test_path, X_test),
        'y_train': (y_train_path, y_train),
        'y_test': (y_test_path, y_test)
    }
    for name, (path, data) in output_paths.items():
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory for {name}: {output_dir}")
        try:
            # Guardar el archivo en la carpeta indicada como name.csv
            save_path = os.path.join(path, f'{name}.csv')
            data.to_csv(save_path, index=False, header=True)
            print(f"{name} saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving {name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Splitting Script")
    parser.add_argument("--encoded_scaled_data_path", type=str, help="Path to the input encoded and scaled CSV dataset", required=True)
    parser.add_argument("--X_train_path", type=str, help="Path to save the training features", required=True)
    parser.add_argument("--X_test_path", type=str, help="Path to save the testing features", required=True)
    parser.add_argument("--y_train_path", type=str, help="Path to save the training labels", required=True)
    parser.add_argument("--y_test_path", type=str, help="Path to save the testing labels", required=True)
    args = parser.parse_args()
    split_data(args.encoded_scaled_data_path, args.X_train_path, args.X_test_path, args.y_train_path, args.y_test_path)
