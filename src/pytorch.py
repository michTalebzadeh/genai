import torch
from torch import nn

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_predicted = self.linear(x)
        return y_predicted

# Create an instance of the model
model = LinearRegression(1, 1)  # Input dim 1 (single feature), output dim 1

# Define some sample data (consider replacing this with your actual data)
x_train = torch.tensor([[1.0], [2.0], [3.0]])  # Shape: (3, 1)
y_train = torch.tensor([[2.0], [4.0], [5.0]])  # Shape: (3, 1)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Choose an optimizer (SGD with a learning rate of 0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (adjust number of epochs for better results)
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train)

    # Backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction for a new input (after training)
new_x = torch.tensor([[4.0]])  # Shape: (1, 1)
y_predicted = model(new_x)

print(f"Predicted value for x = 4.0 after training: {y_predicted.item()}")

