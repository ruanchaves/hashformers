from gpt2_encoder import GPT2Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.hidden_1 = nn.Linear(4, 64)
        self.hidden_2 = nn.Linear(64, 64)
        self.hidden_3 = nn.Linear(64, 1)
        
    def forward(self, input_features):
        x = self.hidden_1(input_features)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return x

def main():

    input_features = []
    labels = []

    model = MLP()

    model = model.to('cuda')

    print(f'The model has {count_parameters(model):,} trainable parameters')

    num_epochs = 2
    loss_function = nn.MSELoss()
    loss_function.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        print("Epoch" + str(epoch + 1))
        train_loss = 0
        for idx, feature in enumerate(input_features):

            model.zero_grad()

            probs = model(feature)
            target = torch.tensor([labels[idx]], dtype=torch.long, device='cuda')

            loss = loss_function(probs, target)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()

    torch.save(model, 'mlp_model.pth')

if __name__ == '__main__':
    main()