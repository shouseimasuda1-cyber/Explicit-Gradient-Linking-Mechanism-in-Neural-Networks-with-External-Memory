import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.colors as colors



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
    """
    カスタムWriteFunctionを適用するためのラッパー関数。
    """
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
        return self.memory, write_weights

    def read_batch(self, x_batch, t):
        query = self.read_query_net(x_batch)
        memory_for_read = self.memory if self.grad_link else self.memory
        if self.grad_link:
            id_token = f"{t}"
            read_result, read_weights = read_with_attention_gradlink(memory_for_read, query, id_token)
        else:
            read_result, read_weights = read_with_attention_no_gradlink(memory_for_read, query)
        return self.read_out_net(read_result), read_weights

def visualize_attention(attention_weights, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto', norm=colors.LogNorm(vmin=1e-5, vmax=attention_weights.max()))
    plt.title(title, fontsize=16)
    plt.xlabel('Memory Slot')
    plt.ylabel('Time Step')
    plt.colorbar(label='Attention Weight')
    plt.show()


def train_model(grad_link=True, epochs=300):
    grad_list.clear()

    input_dim = 80
    mem_dim = 160
    num_slots = 100
    seq_len = 50
    batch_size = 32  
    lr = 3e-4
    clip_value = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionMemoryInterface(input_dim, mem_dim, num_slots, grad_link=grad_link).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_list = []
    total_grad_norm_list = []

    
    vis_epochs = [0, epochs - 1] 

    write_weights_list = []
    read_weights_list = []


    for epoch in range(epochs):
        if grad_link:
            grad_list.clear()

        
        x_seq = torch.randn(batch_size, seq_len, input_dim).to(device)
        model.reset_memory(batch_size, device=device)

        
        epoch_write_weights = []
        for t in range(seq_len):
            _, write_weights = model.write_batch(x_seq[:, t, :], t)
            if epoch in vis_epochs and batch_size > 0:
                 epoch_write_weights.append(write_weights[0].detach().cpu().numpy())
        if epoch in vis_epochs and batch_size > 0:
            write_weights_list.append(np.array(epoch_write_weights))


        
        outputs = []
        epoch_read_weights = []
        for t in range(seq_len):
            out, read_weights = model.read_batch(x_seq[:, t, :], t)
            outputs.append(out.unsqueeze(1))
            if epoch in vis_epochs and batch_size > 0:
                epoch_read_weights.append(read_weights[0].detach().cpu().numpy())
        if epoch in vis_epochs and batch_size > 0:
            read_weights_list.append(np.array(epoch_read_weights))

        outputs = torch.cat(outputs, dim=1)

        loss = criterion(outputs, x_seq)
        optimizer.zero_grad()
        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        total_grad_norm_list.append(total_norm)


        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        loss_list.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Grad Link: {grad_link}")

    
    if len(write_weights_list) == 2:
        vis_epochs_str = [str(e) for e in vis_epochs]
        title_prefix = f"With Gradient Link" if grad_link else "Without Gradient Link"
        visualize_attention(write_weights_list[0], f"{title_prefix}: Write Attention (Epoch {vis_epochs_str[0]})")
        visualize_attention(read_weights_list[0], f"{title_prefix}: Read Attention (Epoch {vis_epochs_str[0]})")
        visualize_attention(write_weights_list[1], f"{title_prefix}: Write Attention (Epoch {vis_epochs_str[1]})")
        visualize_attention(read_weights_list[1], f"{title_prefix}: Read Attention (Epoch {vis_epochs_str[1]})")


    return loss_list, total_grad_norm_list


print("--- Training with Gradient Link ---")
loss_gradlink, grad_norm_gradlink = train_model(grad_link=True)
print("\n--- Training without Gradient Link ---")
loss_nolink, grad_norm_nolink = train_model(grad_link=False)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_gradlink, label="With Gradient Link")
plt.plot(loss_nolink, label="Without Gradient Link")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Copy Task: Loss Comparison")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(grad_norm_gradlink, label="With Gradient Link")
plt.plot(grad_norm_nolink, label="Without Gradient Link")
plt.xlabel("Epoch")
plt.ylabel("Total Gradient Norm")
plt.title("Copy Task: Total Gradient Norm Comparison")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
