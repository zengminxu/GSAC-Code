# train/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.siam_mast import SiamMAST
from data.dataset import VideoDataset  # Custom dataset
from utils.metrics import calculate_accuracy
from config import Config

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    # Step 1: Path 1 - Initialize CNN1 with the pretrained network
    print("Initializing CNN1 with pretrained weights...")
    
    # Load pretrained weights to initialize the model
    # Assuming pretrained weights are loaded elsewhere in the model initialization
    # model.load_pretrained_weights()  # Uncomment this line to load pretrained weights

    # Step 1: Train CNN1 + LSTM1 for M1 iterations
    print("Training CNN1 + LSTM1 for M1 iterations...")
    for epoch in range(num_epochs):
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

    # Step 2: Path 2 - Initialize CNN2 + LSTM2 with weights from CNN1 + LSTM1
    print("Initializing CNN2 + LSTM2 with weights from CNN1 + LSTM1...")
    # This step would typically be handled in the SiamMAST model initialization
    # Here we ensure CNN2 inherits weights from CNN1

    # Step 2: Train CNN2 + LSTM2 with two sequences of frames S and S'
    print("Training CNN2 + LSTM2 with sequence frames...")
    # The training process is similar to the above but on a different dataset
    # Note that you may want to vary inputs and labels for this step

    # Step 3: Fine-tune the spatial motion-awareness sub-network together with CNNx + LSTMx
    print("Fine-tuning spatial motion-awareness sub-network...")
    for _ in range(Config.M2):
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

    # Step 4: Fine-tune the temporal motion-awareness sub-network
    print("Fine-tuning temporal motion-awareness sub-network...")
    for _ in range(Config.M2):
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

    # Step 5: Jointly train the whole network
    print("Jointly training the whole network...")
    for _ in range(Config.M3):  # M3 iterations
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

    print("Training complete.")

if __name__ == "__main__":
    model = SiamMAST(num_classes=Config.NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    train_dataset = VideoDataset()  # Load custom dataset
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Training process
    train(model, train_loader, criterion, optimizer, Config.NUM_EPOCHS)
