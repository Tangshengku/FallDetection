import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Network Building
class FallDetectionModel(nn.Module):
    def __init__(self):
        super(FallDetectionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)  
    
    def forward(self, x):
        out, (_, _) = self.lstm(x)
        x = self.fc(out)
        return torch.sigmoid(x)

# Model Training Function
def train_model(model, X_train, y_train, epochs):
    model.train()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train, dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def data_prepare(data_path, test_size=0.2):
    # Data Loading
    data = pd.DataFrame()
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        files = os.listdir(data_path)
        if len(files) == 0:
            RuntimeError("Invalid data path! Please give the right path to dataset.\
                         It can be the directory to csv or csv file path.")
        data_lists = []
        for file_name in files:
            data_per_file = pd.read_csv(os.path.join(data_path,file_name), header=None)
            data_lists.append(data_per_file)
        data = pd.concat(data_lists, axis=0, join='inner')
        data.reset_index(drop=True, inplace=True)

    X = data.iloc[:, :-1].values  # Input
    y = data.iloc[:, -1].values  # Labels

    # Data Pre-Processing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training and Testing Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='FallDetection',
                    description='Train model to detect fall')
    parser.add_argument("--data_path", type=str, default="/Users/bryson/Desktop/claude_data/data/", help="Data path for training")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test data size for evaluation")
    parser.add_argument("--lr", type=float, default=0.001, help="lr for model training")
    parser.add_argument("--epoch", type=float, default=200, help="epoch for model training")

    args = parser.parse_args()


    X_train, X_test, y_train, y_test = data_prepare(args.data_path, args.test_size)
    
    model = FallDetectionModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, X_train, y_train, args.epoch)

    # Evaluation
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        labels = torch.tensor(y_test, dtype=torch.float32)
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        accuracy = (predicted == labels).float().mean()
        print(f'Accuracy: {accuracy.item()}')