import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd


# 定义模型类
class BertForSentimentAnalysis(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_size=768, dropout_prob=0.01):
        super(BertForSentimentAnalysis, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        next(fp)  # Skip header
        for line in fp.readlines():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                data.append({'text': parts[1], 'label': int(parts[0])})
            elif len(parts) == 3:
                data.append({'text': parts[2], 'label': int(parts[1])})
    return data


# 定义数据集类
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.print_sample_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['text']
        label = self.data[index]['label']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def print_sample_data(self):
        print(f"Loaded {len(self.data)} samples.")
        for i, sample in enumerate(self.data[:5]):
            print(f"Sample {i + 1}: {sample}")


dataset_name = 'ChnSentiCorp'

# 加载数据
train_data = load_data(f'./data/{dataset_name}/train.tsv')
dev_data = load_data(f'./data/{dataset_name}/dev.tsv')
test_data = load_data(f'./data/{dataset_name}/test.tsv')

# 初始化BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert_model/bert-base-chinese')
max_len = 140

# 创建数据集实例
train_dataset = SentimentDataset(data=train_data, tokenizer=tokenizer, max_len=max_len)
dev_dataset = SentimentDataset(data=dev_data, tokenizer=tokenizer, max_len=max_len)
test_dataset = SentimentDataset(data=test_data, tokenizer=tokenizer, max_len=max_len)
# 创建数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义训练和评估函数
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=8, device='cuda'):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
        evaluate_model(model, dev_loader, device)
    print("Training complete.")
    predict_and_save_results(model, test_loader, device, output_file=f'{dataset_name}.tsv')


def evaluate_model(model, data_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")


def predict_and_save_results(model, data_loader, device, output_file):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # 保存预测结果到文件
    with open(output_file, 'w') as f:
        f.write("index\tprediction\n")
        for idx, prediction in enumerate(predictions):
            f.write(f"{idx}\t{prediction}\n")
    print(f"Results saved to {output_file}")


# 加载模型和定义损失函数及优化器
model = BertForSentimentAnalysis(bert_model='bert_model/bert-base-chinese', num_labels=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 训练和评估模型
train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=6)
