import torch
from torch.utils.data import DataLoader
from model.siam_mast import SiamMAST
from data.dataset import VideoDataset 
from utils.metrics import calculate_accuracy

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 计算并打印准确率
        accuracy = calculate_accuracy(outputs, labels)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    model = SiamMAST()

    # 选择损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = VideoDataset()  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练过程
    num_epochs = 10
    train(model, train_loader, criterion, optimizer, num_epochs)
