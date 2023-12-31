import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import argparse
from DisasterDataset import DisasterDataset
from DisasterClassifier import DisasterClassifier
from tqdm import tqdm
import os

def main():
    '''
    Run in terminal with: 
    cd NLP/UT-NLP-Project/src/
    python main.py --expr_id=id
    '''

    # load params

    parser = argparse.ArgumentParser(description='Test Argparse')
    
    parser.add_argument('--file_path', type=str, default=r'../datasets/')
    parser.add_argument('--train_file', type=str, default=r'train.csv')
    parser.add_argument('--test_file', type=str, default=r'test.csv')
    parser.add_argument('--bert_out', type=int, default=64)
    parser.add_argument('--adapter', type=str, default='mrpc')
    parser.add_argument('--adapter_config', type=str, default='pfeiffer')
    parser.add_argument('--dime_reduction_layer_out', type=int, default=32)
    parser.add_argument('--hidden_layer_1', type=int, default=16)
    parser.add_argument('--hidden_layer_2', type=int, default=8)
    parser.add_argument('--drop_out_rate', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--expr_id', type=int)
    
    args = parser.parse_args()

    # parse paramters
    file_path = args.file_path
    train_file = args.train_file
    test_file = args.test_file
    bert_out = args.bert_out
    adapter = args.adapter
    adapter_config = args.adapter_config
    dime_reduction_layer_out = args.dime_reduction_layer_out
    hidden_layer_1 = args.hidden_layer_1
    hidden_layer_2 = args.hidden_layer_2
    drop_out_rate = args.drop_out_rate
    EXPERIMENT_ID = args.expr_id
    EPOCH = args.epoch


    # define device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('--- gpu ---')
    else:
        device = torch.device("cpu")
        print('--- cpu ---')
    
    
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
    train_data, temp_data, train_labels, temp_labels = train_test_split(raw_train_data['text'], raw_train_data['target'], test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)


    # convert data into tensors
    train_data = torch.LongTensor(train_data.tolist()).to(device)
    train_labels = torch.LongTensor(train_labels.tolist()).to(device)
    val_data = torch.LongTensor(val_data.tolist()).to(device)
    val_labels = torch.LongTensor(val_labels.tolist()).to(device)
    test_data = torch.LongTensor(test_data.tolist()).to(device)
    test_labels = torch.LongTensor(test_labels.tolist()).to(device)
    
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
                               hidden_layer_1, hidden_layer_2, drop_out_rate).to(device)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    if os.path.exists(path_checkpoint):
        print('load model...')
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch


    # train
    iepoches = []
    f1_vals = []
    PATIENCE = 5
    best_f1 = 0.0
    epochs_since_best = 0
    
    # Training loop
    for iepoch in range(start_epoch + 1, EPOCH):
        print(f'train epoch {iepoch}/{EPOCH}')
        model.train()
        for data, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        # Validation and test

        model.eval()

        # validate
        val_predicted_labels = []
        val_true_labels = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_predicted_labels.extend(predicted.tolist())
                val_true_labels.extend(labels.tolist())

        f1 = f1_score(val_true_labels, val_predicted_labels, average='macro')

        iepoches.append(iepoch)
        f1_vals.append(f1)
        print(f'Validation F1 Score: {f1 * 100:.2f}%')

        # Check for early stopping
        if f1 > best_f1:
            best_f1 = f1
            epochs_since_best = 0

            # save model
            checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": iepoch
                }
            if not os.path.isdir("../model/checkpoint"):
                os.mkdir("../model/checkpoint")
            torch.save(checkpoint, f'../model/checkpoint/ckpt_expr_{EXPERIMENT_ID}.pth')

        else:
            epochs_since_best += 1

        if epochs_since_best >= PATIENCE:
            print("Early stopping!")
            break
            
            # test
        if iepoch % 2 == 0:
            test_predicted_labels = []
            test_true_labels = []
            with torch.no_grad():
                for data, labels in tqdm(test_loader):
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    test_predicted_labels.extend(predicted.tolist())
                    test_true_labels.extend(labels.tolist())
                    
            f1 = f1_score(test_true_labels, test_predicted_labels, average='macro')
            
            iepoches.append(iepoch)
            f1_vals.append(f1)
            print(f'Test F1 Score: {f1 * 100:.2f}%')


if __name__ == "__main__":
    main()