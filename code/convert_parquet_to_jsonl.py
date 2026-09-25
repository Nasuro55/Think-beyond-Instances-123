import pandas as pd
import os

def convert_parquet_to_jsonl(input_file, output_file):
    print(f"Reading {input_file} ...")
    
    try:
        # 1. Read parquet file
        df = pd.read_parquet(input_file, engine='pyarrow')
        
        print(f"Read successful, total {len(df)} rows. Writing to {output_file} ...")
        
        # 2. Write jsonl file
        # orient='records': Convert each row to a dictionary
        # lines=True: Enable jsonl mode (newline separated)
        # force_ascii=False: Prevent Chinese or special math symbols from becoming garbled
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print("Conversion complete!")
        
    except Exception as e:
        print(f"Conversion error: {e}")

if __name__ == "__main__":
    # Set input and output filenames
    input_path = "train-9k.parquet"
    output_path = "train-9k.jsonl"
    
    if os.path.exists(input_path):
        convert_parquet_to_jsonl(input_path, output_path)
    else:
        print(f"Error: File not found {input_path}")