
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from diffusers.models.normalization import FP32LayerNorm

class QueryAttention(nn.Module):
    """
    Query-based attention pooling module using PyTorch's built-in MultiheadAttention.
    Uses learnable query vectors to attend to sequence features.
    """
    def __init__(self, feature_dim, num_queries=1, num_heads=8, dropout=0.1, layer_norm=False, return_type=None, product_text=False, text_dim=768):
        super(QueryAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.return_type = return_type
        self.product_text = product_text

        # Use PyTorch's built-in MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        # Learnable query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, feature_dim))
        # Initialize query parameters
        nn.init.xavier_uniform_(self.queries)

        if self.layer_norm:
            self.norm = FP32LayerNorm(feature_dim, eps=1e-6, elementwise_affine=False)
        
        if self.product_text:
            self.text_proj = nn.Linear(text_dim, feature_dim)
            nn.init.xavier_uniform_(self.text_proj.weight)
            if self.text_proj.bias is not None:
                nn.init.zeros_(self.text_proj.bias)

        
    def forward(self, x, e = None, text = None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
        Returns:
            Pooled features of shape [batch_size, feature_dim]
        """

        if self.layer_norm:
            x = self.norm(x)

        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.shape) == 2:  # [batch_size, feature_dim]
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
            seq_len = 1
        elif len(x.shape) == 3:  # [batch_size, seq_len, feature_dim]
            seq_len = x.shape[1]
        elif len(x.shape) == 4:  # [sp_size, batch_size, seq_len, feature_dim]
            # Handle sequence parallel case
            sp_size, batch_size, seq_len, feature_dim = x.shape
            x = x.view(sp_size * batch_size, seq_len, feature_dim)
            batch_size = sp_size * batch_size
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Expand queries to batch size
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_queries, feature_dim]
        if e is not None:
            queries = queries + e.unsqueeze(0).expand(batch_size, -1, -1)
        # Use PyTorch's MultiheadAttention
        # query: [batch_size, num_queries, feature_dim]
        # key, value: [batch_size, seq_len, feature_dim]
        attended, attention_weights = self.multihead_attn(
            query=queries,
            key=x,
            value=x,
            need_weights=False  # We don't need attention weights for pooling
        )
        
        # attended: [batch_size, num_queries, feature_dim]
        
        # If multiple queries, average them
        if self.num_queries > 1:
            output = attended.mean(dim=1)  # [batch_size, feature_dim]
        else:
            output = attended.squeeze(1)  # [batch_size, feature_dim]
        
        # Handle sequence parallel case
        if len(original_shape) == 4:
            output = output.view(sp_size, batch_size // sp_size, -1)
            output = output.mean(dim=0)  # Average across SP devices
        
        if self.layer_norm:
            output = self.norm(output)

        if self.return_type == 'query':
            output = output + queries

        if self.product_text and text is not None:
            output_product_text = torch.mul(self.text_proj(text), output)
            return output_product_text
        else:
            return output

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # First hidden layer
        self.fc2 = nn.Linear(1024, 512)        # Second hidden layer
        self.fc3 = nn.Linear(512, 1)           # Output layer (binary classification)
        
        # 初始化权重，避免梯度消失
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 注意：这里不应用sigmoid，因为forward_siamese会处理
        return x

class MultiHead(nn.Module):
    def __init__(self, input_dim, num_heads = 3):
        super().__init__()
        self.num_heads = num_heads
        self.mlps = torch.nn.ModuleList(
            [MLP(input_dim) for _ in range(num_heads)]
        )

    def forward_mlp(self, head_idx, x):
        return torch.sigmoid(self.mlps[head_idx](x))

    def forward(self, x):
        out = [self.forward_mlp(h, x) for h in range(self.num_heads)]
        return torch.stack(out)

def forward_mlp(model, input):
    return torch.sigmoid(model(input))

def forward_siamese(model, input1, input2):
    # Pass both inputs through the same model (weight sharing)
    reward1 = model(input1)
    reward2 = model(input2)
    # Compute the difference between the two embeddings
    diff = reward1 - reward2

    # Use this difference for binary prediction (preference/ranking)
    return torch.sigmoid(diff)

def train_model(model, device, model_mode, X_train, y_train, X_test, y_test, epochs=3, lr=0.001, batch_size = 512, verbose=False, ealry_stopping_patience=3):
    model = model.to(device)  # Move model to GPU
    criterion = nn.BCELoss()  # Binary cross-entropy loss with logits for numerical stability
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = min(batch_size, X_train.shape[0])
    val_losses = []
    for epoch in range(epochs):
        for n_batch in range(0, X_train.shape[0], batch_size):

            # randomly selecte batch
            batch_idx = torch.randperm(X_train.shape[0])[:batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            if model_mode == 'clf':
                outputs = forward_mlp(model, X_batch)
            elif model_mode == 'siamese':
                outputs = forward_siamese(model, X_batch[:, 0], X_batch[:, 1])
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            if model_mode == 'clf':
                val_outputs = forward_mlp(model, X_test)
            elif model_mode == 'siamese':
                val_outputs = forward_siamese(model, X_test[:, 0], X_test[:, 1])
        # early stopping?
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.cpu().detach().item())
        if len(val_losses) > ealry_stopping_patience:
            if all(val_losses[-1] > x for x in val_losses[-(ealry_stopping_patience+1):-1]):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.cpu().detach().item()}, Val Loss: {val_loss.cpu().detach().item()}")
        # accuracy
        val_outputs = val_outputs.cpu().detach().numpy()
        val_pred = (val_outputs > 0.5).astype(int)
        accuracy = (val_pred == y_test.cpu().detach().numpy()).mean()
        if verbose:
            print(f"Accuracy: {accuracy}")

def save_model(model, path):
    torch.save(model.state_dict(), path)