import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Mean and log-variance for weight distribution
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_log_var = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Mean and log-variance for bias distribution
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Sample weights and biases
        w_std = torch.exp(0.5 * self.w_log_var)
        b_std = torch.exp(0.5 * self.b_log_var)
        
        w = self.w_mu + w_std * torch.randn_like(w_std)
        b = self.b_mu + b_std * torch.randn_like(b_std)
        
        return F.linear(x, w, b)

# Bayesian Neural Network Model
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.blinear1(x))
        return self.blinear2(x)

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    X = torch.randn(100, 2)
    y = (X[:, 0] > X[:, 1]).long()  # Simple classification task
    
    model = BayesianNN(input_dim=2, hidden_dim=10, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
