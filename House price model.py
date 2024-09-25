import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load dataset
# Assuming you have a CSV file with features: 'area', 'income', 'taxes', 'location' (categorical or coordinates), etc.
# and a target: 'price' for the house price.
data = pd.read_csv('indiana_housing.csv')

# Preprocess the data
X = data[['area', 'income', 'taxes', 'location']].values
y = data['price'].values

# If 'location' is categorical, use one-hot encoding or encode it numerically
X = pd.get_dummies(data[['area', 'income', 'taxes', 'location']], drop_first=True).values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple neural network model
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
model = HousePricePredictor(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Predicting house price for a new area in Indiana
def predict_price(new_data):
    new_data = torch.tensor(scaler.transform([new_data]), dtype=torch.float32)
    with torch.no_grad():
        predicted_price = model(new_data).item()
    return predicted_price

# Example: predicting price for a new house with given features
# Replace these values with actual features
new_house = [2000, 50000, 3000, 1]  # example: area, income, taxes, location (encoded)
predicted_price = predict_price(new_house)
print(f'Predicted Price for new house: ${predicted_price:.2f}')
