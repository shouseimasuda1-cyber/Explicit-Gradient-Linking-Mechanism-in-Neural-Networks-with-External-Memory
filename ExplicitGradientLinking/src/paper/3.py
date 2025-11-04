"""
Run all experiments for the paper:
"Explicit Gradient Linking Mechanism in Neural Networks with External Memory"
Author: Shousei Masuda (2025)

Usage:
    python run_all_experiments.py
Results will be saved under `figures/` and `results/`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



grad_list = {}


def read_with_attention_no_gradlink(memory_tensor, query):
    attn_logits = torch.bmm(memory_tensor, query.unsqueeze(2)).squeeze(2)
    attn_weights = F.softmax(attn_logits, dim=1)
    read_output = torch.bmm(attn_weights.unsqueeze(1), memory_tensor).squeeze(1)
    return read_output, attn_weights


def read_with_attention_gradlink(memory_tensor, query, id_token):
    attn_logits = torch.bmm(memory_tensor, query.unsqueeze(2)).squeeze(2)
    attn_weights = F.softmax(attn_logits, dim=1)
    read_output = torch.bmm(attn_weights.unsqueeze(1), memory_tensor).squeeze(1)

    def save_grad_hook(grad):
        if id_token in grad_list:
            grad_list[id_token] += grad
        else:
            grad_list[id_token] = grad
    read_output.register_hook(save_grad_hook)
    return read_output, attn_weights


class WriteFunctionGradLink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, id_token, memory_tensor, write_weights):
        ctx.save_for_backward(input_tensor, write_weights, memory_tensor)
        ctx.id_token = id_token

        write_weights_expanded = write_weights.unsqueeze(2)
        input_tensor_expanded = input_tensor.unsqueeze(1)

        
        memory_out = memory_tensor.clone()
        memory_out = memory_out * (1 - write_weights_expanded) + input_tensor_expanded * write_weights_expanded
        return memory_out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, write_weights, memory_tensor = ctx.saved_tensors
        id_token = ctx.id_token

        B, N, D = memory_tensor.shape

        grad_read = grad_list.pop(id_token, None)

        
        grad_memory = grad_output.clone()

        
        if grad_read is not None:
            grad_memory = grad_memory + grad_read.unsqueeze(1)

        
        
        grad_input = torch.sum(write_weights.unsqueeze(2) * grad_memory, dim=1)

        
        grad_weights = torch.sum(
            grad_memory * (input_tensor.unsqueeze(1) - memory_tensor), dim=2
        )

        
        grad_memory_final = grad_memory * (1 - write_weights.unsqueeze(2))

        return grad_input, None, grad_memory_final, grad_weights


def write_to_memory_gradlink(input_tensor, id_token, memory_tensor, write_weights):
    return WriteFunctionGradLink.apply(input_tensor, id_token, memory_tensor, write_weights)


class AttentionMemoryInterface(nn.Module):
    def __init__(self, input_dim, mem_dim, num_slots, grad_link=True):
        super().__init__()
        self.write_net = nn.Linear(input_dim, mem_dim)
        self.write_query_net = nn.Linear(input_dim, mem_dim)
        self.read_query_net = nn.Linear(input_dim, mem_dim)
        self.read_out_net = nn.Linear(mem_dim, input_dim)
        self.num_slots = num_slots
        self.mem_dim = mem_dim
        self.memory = None
        self.grad_link = grad_link

    def reset_memory(self, batch_size, device='cpu'):
        self.memory = torch.randn(batch_size, self.num_slots, self.mem_dim, device=device)

    def write_batch(self, x_batch, t):
        mem_values = self.write_net(x_batch)
        write_query = self.write_query_net(x_batch)
        write_logits = torch.bmm(self.memory, write_query.unsqueeze(2)).squeeze(2)
        write_weights = F.softmax(write_logits, dim=1)
        if self.grad_link:
            id_token = f"{t}"
            self.memory = write_to_memory_gradlink(mem_values, id_token, self.memory, write_weights)
        else:
            
            write_weights_expanded = write_weights.unsqueeze(2)
            input_tensor_expanded = mem_values.unsqueeze(1)
            self.memory = self.memory * (1 - write_weights_expanded) + input_tensor_expanded * write_weights_expanded
        return self.memory

    def read_batch(self, x_batch, t):
        query = self.read_query_net(x_batch)
        memory_for_read = self.memory if self.grad_link else self.memory
        if self.grad_link:
            id_token = f"{t}"
            read_result, _ = read_with_attention_gradlink(memory_for_read, query, id_token)
        else:
            read_result, _ = read_with_attention_no_gradlink(memory_for_read, query)
        return self.read_out_net(read_result)


def train_model(grad_link=True, epochs=300):
    grad_list.clear()

    input_dim = 80
    mem_dim = 160
    num_slots = 100
    seq_len = 50
    batch_size = 32  
    lr = 5e-4
    clip_value = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionMemoryInterface(input_dim, mem_dim, num_slots, grad_link=grad_link).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_list = []

    for epoch in range(1,epochs+1):
        if grad_link:
            grad_list.clear()

        
        x_seq = torch.randn(batch_size, seq_len, input_dim).to(device)
        model.reset_memory(batch_size, device=device)

        
        for t in range(seq_len):
            model.write_batch(x_seq[:, t, :], t)

        
        outputs = []
        for t in range(seq_len):
            outputs.append(model.read_batch(x_seq[:, t, :], t).unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)

        loss = criterion(outputs, x_seq)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        loss_list.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Grad Link: {grad_link}")

    return loss_list


print("--- Training with Gradient Link ---")
loss_gradlink = train_model(grad_link=True)
print("\n--- Training without Gradient Link ---")
loss_nolink = train_model(grad_link=False)

plt.figure(figsize=(10,6))
plt.plot(loss_gradlink, label="With Gradient Link")
plt.plot(loss_nolink, label="Without Gradient Link")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Copy Task: Gradient Link vs No Link")
plt.legend()
plt.grid(True)
plt.show()
