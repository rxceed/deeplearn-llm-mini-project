import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_csv(input_path, output_dir, test_size=0.2, random_state=42):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Split data
    print(f"Splitting data (test_size={test_size})...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"Saving splits to {output_dir}...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Original size: {len(df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

if __name__ == "__main__":
    input_file = "dataset/learning-agency-lab-automated-essay-scoring-2/train.csv"
    output_directory = "dataset/split"
    split_csv(input_file, output_directory)
