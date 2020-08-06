import torch


class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.dense1=torch.nn.Linear(128,64,bias=True)
        self.relu=torch.nn.ReLU()
        self.batch_norm1=torch.nn.BatchNorm1d(64)
        self.dense2 = torch.nn.Linear(64, 32, bias=True)
        self.batch_norm2 = torch.nn.BatchNorm1d(32)
        self.dense3 = torch.nn.Linear(32, 1, bias=True)

    def forward(self,x):
        x = self.dense1(x)
        x=self.relu(x)
        x = self.batch_norm1(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.dense3(x)
        return x