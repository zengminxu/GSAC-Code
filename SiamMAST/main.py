from train.train import train
from model.siam_mast import SiamMAST

def main():
    model = SiamMAST()


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    train_dataset = VideoDataset()  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


    num_epochs = 10
    train(model, train_loader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()
