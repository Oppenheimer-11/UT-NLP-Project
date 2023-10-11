import argparse
import pandas as pd

def main():
    '''
    Run with: python src/main.py
    '''

    parser = argparse.ArgumentParser(description='Test Argparse')
    
    parser.add_argument('--file_path', type=str, default=r'./datasets/')
    parser.add_argument('--train_file', type=str, default=r'train.csv')
    parser.add_argument('--test_file', type=str, default=r'test.csv')
    parser.add_argument('--subm_file', type=str, default=r'sample_submission.csv')
    
    args = parser.parse_args()

    file_path = args.file_path
    train_file = args.train_file
    test_file = args.test_file
    subm_file = args.subm_file

    raw_train_data = pd.read_csv(file_path+train_file)
    
    print(raw_train_data.shape)
    
if __name__ == "__main__":
    main()