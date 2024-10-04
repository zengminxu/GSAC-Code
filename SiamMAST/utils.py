# config.py

class Config:
    # Configuration settings for the training
    NUM_CLASSES = 10  # Set the number of action classes
    BATCH_SIZE = 64  # Set the batch size for training
    LEARNING_RATE = 1e-3  # Set the initial learning rate
    NUM_EPOCHS = 10  # Set the number of epochs for training
    M1 = 5000  # Number of iterations for Path 1
    M2 = 5000  # Number of iterations for fine-tuning
    M3 = 10000  # Number of iterations for joint training
