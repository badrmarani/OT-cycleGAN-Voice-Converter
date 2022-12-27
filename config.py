import torch

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0001
batch_size = 32
epochs = 10
lambda_id = 0.0
lambda_cyc = 0.0
