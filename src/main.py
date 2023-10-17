import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from DisasterDataset import DisasterDataset
from DisasterClassifier import DisasterClassifier
import tqdm
import os

def main():
    '''
    Run with: 
    cd /src
    python src/main.py
    '''

    parser = argparse.ArgumentParser(description='Test Argparse')
    
    parser.add_argument('--file_path', type=str, default=r'../datasets/')
    parser.add_argument('--train_file', type=str, default=r'train.csv')
    parser.add_argument('--test_file', type=str, default=r'test.csv')
    parser.add_argument('bert_out', type=int, default=32)
    parser.add_argument('adapter', type=str, default='mrpc')
    parser.add_argument('adapter_config', type=str, default='pfeiffer')
    parser.add_argument('dime_reduction_layer_out', type=int, default=64)
    parser.add_argument('hidden_lay_1', type=int, default=32)
    parser.add_argument('hidden_lay_2', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--expr_id', type=int)

    # parser.add_argument('--subm_file', type=str, default=r'sample_submission.csv')
    
    args = parser.parse_args()

    # parse paramters
    file_path = args.file_path
    train_file = args.train_file
    test_file = args.test_file
    EXPERIMENT_ID = args.expr_id
    EPOCH = args.epoch

    

    # load files
    raw_train_data = pd.read_csv(file_path+train_file)
    raw_test_data = pd.read_csv(file_path+test_file)
    

    # Tokenize the text using BERT's tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the text using BERT's tokenizer and ensure uniform sequence length
    max_sequence_length = 128  # Adjust to an appropriate sequence length

    raw_train_data['text'] = raw_train_data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_sequence_length, truncation=True, padding='max_length'))
    raw_test_data['text'] = raw_test_data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_sequence_length, truncation=True, padding='max_length'))
    raw_train_data['text'].head()
    

    # Split the data into train and train sets
    # train : val : test = 70% : 15% : 15%
    train_data, test_data, train_labels, test_labels = train_test_split(raw_train_data['text'], raw_train_data['target'], test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(raw_train_data['text'], raw_train_data['target'], test_size=0.5, random_state=42)

    # convert data into tensors
    train_data = torch.LongTensor(train_data.tolist())
    train_labels = torch.LongTensor(train_labels.tolist())
    val_data = torch.LongTensor(val_data.tolist())
    val_labels = torch.LongTensor(val_labels.tolist())
    test_data = torch.LongTensor(test_data.tolist())
    test_labels = torch.LongTensor(test_labels.tolist())
    
    # create data loader
    train_dataset = DisasterDataset(train_data, train_labels)
    val_dataset = DisasterDataset(val_data, val_labels)
    test_dataset = DisasterDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)


    # load model

    path_checkpoint = f'../model/checkpoint/ckpt_expr_{EXPERIMENT_ID}.pth'

    start_epoch = -1

    model = DisasterClassifier(bert_out, adapter, adapter_config, dime_reduction_layer_out, 
                               hidden_lay_1, hidden_lay_2)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(path_checkpoint):
        print('load model...')
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch


    # train
    iepoches = []
    f1_vals = []

    # Training loop
    for iepoch in range(start_epoch + 1 ,EPOCH):
        print(f'train epoch {iepoch}/{EPOCH}')
        model.train()
        for data, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # save model
        checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": iepoch
            }
        if not os.path.isdir("../model/checkpoint"):
            os.mkdir("../model/checkpoint")
        torch.save(checkpoint, f'../model/checkpoint/ckpt_expr_{EXPERIMENT_ID}.pth')
            
        # Validation loop
        if iepoch % 2 == 0:
            model.eval()
            
            # validate
            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for data, labels in tqdm(val_loader):
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())
                    
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            
            iepoches.append(iepoch)
            f1_vals.append(f1)
            print(f'Validation F1 Score: {f1 * 100:.2f}%')
            
            # test
            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for data, labels in tqdm(test_loader):
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_labels.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())
                    
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            
            iepoches.append(iepoch)
            f1_vals.append(f1)
            print(f'Test F1 Score: {f1 * 100:.2f}%')


if __name__ == "__main__":
    main()