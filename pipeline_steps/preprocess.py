import pandas as pd
import os

def preprocess(input_data_path, output_train_path, output_test_path, test_size=0.2, random_state=42):
    """
    Loads the input CSV data, splits into train/test, and saves as new files.
    Args:
        input_data_path (str): local path to input CSV (given by SageMaker)
        output_train_path (str): local path to write train split
        output_test_path (str): local path to write test split
        test_size (float): ratio for test split
        random_state (int): random seed for reproducibility
    """
    print(f"Reading data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    print(f"Loaded data, {len(df)} rows.")
    
    # Basic preprocessing: shuffle & split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_count = int(len(df) * test_size)
    test_df = df.iloc[:test_count]
    train_df = df.iloc[test_count:]

    print(f"Saving train data ({len(train_df)} rows) to {output_train_path}...")
    train_df.to_csv(output_train_path, index=False)
    print(f"Saving test data ({len(test_df)} rows) to {output_test_path}...")
    test_df.to_csv(output_test_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--output-train', type=str, required=True)
    parser.add_argument('--output-test', type=str, required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    preprocess(args.input_data, args.output_train, args.output_test, args.test_size)
