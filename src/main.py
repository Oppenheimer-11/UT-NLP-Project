import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import DisasterDataset

def main():
    '''
    Run with: python src/main.py
    '''

    parser = argparse.ArgumentParser(description='Test Argparse')
    
    parser.add_argument('--file_path', type=str, default=r'./datasets/')
    parser.add_argument('--train_file', type=str, default=r'train.csv')
    parser.add_argument('--test_file', type=str, default=r'test.csv')
    # parser.add_argument('--subm_file', type=str, default=r'sample_submission.csv')
    
    args = parser.parse_args()

    # load data
    file_path = args.file_path
    train_file = args.train_file
    test_file = args.test_file
    # subm_file = args.subm_file
    raw_train_data = pd.read_csv(file_path+train_file)
    raw_test_data = pd.read_csv(file_path+test_file)
    
    # Tokenize the text using BERT's tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the text using BERT's tokenizer and ensure uniform sequence length
    max_sequence_length = 128  
    raw_train_data['text'] = raw_train_data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_sequence_length, truncation=True, padding='max_length'))
    raw_test_data['text'] = raw_test_data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_sequence_length, truncation=True, padding='max_length'))
    
    # Split the data into train and train sets
    # train : val : test = 70% : 15% : 15%
    train_data, test_data, train_labels, test_labels = train_test_split(raw_train_data['text'], raw_train_data['target'], test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(raw_train_data['text'], raw_train_data['target'], test_size=0.5, random_state=42)

    # convert data into Tonsor
    train_data = torch.LongTensor(train_data.tolist())
    train_labels = torch.LongTensor(train_labels.tolist())
    val_data = torch.LongTensor(val_data.tolist())
    val_labels = torch.LongTensor(val_labels.tolist())
    test_data = torch.LongTensor(raw_test_data['text'])
    
    # create data loader
    train_dataset = DisasterDataset(train_data, train_labels)
    val_dataset = DisasterDataset(val_data, val_labels)
    test_dataset = DisasterDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    val_loader = DataLoader(train_dataset, batch_size=32)

if __name__ == "__main__":
    main()