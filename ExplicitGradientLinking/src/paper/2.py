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
import numpy as np
import copy
import matplotlib.colors as colors




grad_list = {}


def read_with_attention(memory_tensor, query, id_token, mode):
    B, N, D = memory_tensor.shape

    attn_logits = torch.bmm(memory_tensor, query.unsqueeze(2)).squeeze(2)
    
    
    attn_weights = F.softmax(attn_logits, dim=1)

    read_output = torch.bmm(attn_weights.unsqueeze(1), memory_tensor).squeeze(1)

    def save_grad_hook(grad):
        g = grad.detach()
        if mode == 'add' and id_token in grad_list:
            grad_list[id_token] = grad_list[id_token] + g
        else:
            grad_list[id_token] = g


    read_output.register_hook(save_grad_hook)
    return read_output, attn_weights


class WriteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, id_token, memory_tensor, write_weights, with_gradient_link, mode):
        ctx.save_for_backward(input_tensor, write_weights, memory_tensor)
        ctx.id_token = id_token
        ctx.with_gradient_link = with_gradient_link
        ctx.mode = mode

        memory_out = memory_tensor.clone()
        w = write_weights.unsqueeze(2)
        x = input_tensor.unsqueeze(1)
        memory_out = memory_out * (1 - w) + x * w
        return memory_out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, write_weights, memory_tensor = ctx.saved_tensors
        id_token = ctx.id_token
        with_gradient_link = ctx.with_gradient_link

        
        grad_memory = grad_output.clone()

        
        if with_gradient_link:
            grad_read = grad_list.pop(id_token, None)
            if grad_read is not None:
                grad_memory = grad_memory + grad_read.detach().unsqueeze(1)

        
        grad_input = torch.sum(write_weights.unsqueeze(2) * grad_memory, dim=1)  
        grad_weights = torch.sum(
            grad_memory * (input_tensor.unsqueeze(1) - memory_tensor),
            dim=2
        )  
        grad_memory_final = grad_memory * (1 - write_weights.unsqueeze(2))  

        return grad_input, None, grad_memory_final, grad_weights, None, None

def write_to_memory(input_tensor, id_token, memory_tensor, write_weights, with_gradient_link, mode):
    return WriteFunction.apply(input_tensor, id_token, memory_tensor, write_weights, with_gradient_link, mode)


class AttentionMemoryInterface(nn.Module):
    def __init__(self, input_dim, mem_dim, num_slots):
        super().__init__()
        self.write_net = nn.Linear(input_dim, mem_dim)
        self.write_query_net = nn.Linear(input_dim, mem_dim)
        self.read_query_net = nn.Linear(input_dim, mem_dim)
        self.read_out_net = nn.Linear(mem_dim, input_dim)
        self.num_slots = num_slots
        self.mem_dim = mem_dim
        self.memory = None

    def reset_memory(self, batch_size, device='cpu'):
        self.memory = torch.randn(batch_size, self.num_slots, self.mem_dim, device=device)

    def write_batch(self, x_batch, t, with_gradient_link, mode):
        mem_values = self.write_net(x_batch)

        write_query = self.write_query_net(x_batch)
        write_logits = torch.bmm(self.memory, write_query.unsqueeze(2)).squeeze(2)
        write_weights = F.softmax(write_logits, dim=1)

        id_token = f"{t}"
        self.memory = write_to_memory(mem_values, id_token, self.memory, write_weights, with_gradient_link, mode)
        return self.memory, write_weights

    def read_batch(self, x_batch, t, mode):
        query = self.read_query_net(x_batch)
        id_token = f"{t}"
        read_result, read_weights = read_with_attention(self.memory, query, id_token, mode)
        return self.read_out_net(read_result), read_weights

def visualize_attention(attention_weights, title):
    """
    ヒートマップを生成して表示する関数。
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto', norm=colors.LogNorm(vmin=1e-5, vmax=attention_weights.max()))
    plt.title(title, fontsize=16)
    plt.xlabel('Memory Slot')
    plt.ylabel('Time Step')
    plt.colorbar(label='Attention Weight')
    plt.show()


def train_and_visualize(with_gradient_link, epochs=500, mode='overwrite'):
    input_dim = 80
    mem_dim = 160
    num_slots = 100
    seq_len = 50
    batch_size = 32  
    lr = 5e-4
    clip_value = 1.0
    epochs = 300

    model = AttentionMemoryInterface(input_dim, mem_dim, num_slots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Start training (with_gradient_link={with_gradient_link}, mode={mode})")

    write_weights_list = []
    read_weights_list = []

    
    
    vis_epochs = [10, 499]
    loss_history = []

    for epoch in range(epochs):
        grad_list.clear()
        
        x_seq = torch.randn(batch_size, seq_len, input_dim).to(device)
        model.reset_memory(batch_size, device=device)

        
        epoch_write_weights = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :] 
            _, write_weights = model.write_batch(x_t, t, with_gradient_link, mode)
            if epoch in vis_epochs and batch_size > 0: 
                epoch_write_weights.append(write_weights[0].detach().cpu().numpy())
        if epoch in vis_epochs and batch_size > 0:
            write_weights_list.append(np.array(epoch_write_weights))

        outputs = []
        epoch_read_weights = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :] 
            out, read_weights = model.read_batch(x_t, t, mode)
            outputs.append(out.unsqueeze(1))
            if epoch in vis_epochs and batch_size > 0: 
                epoch_read_weights.append(read_weights[0].detach().cpu().numpy())
        if epoch in vis_epochs and batch_size > 0:
            read_weights_list.append(np.array(epoch_read_weights))

        outputs = torch.cat(outputs, dim=1)
        loss = criterion(outputs, x_seq)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    
    
    if len(write_weights_list) == 2:
        vis_epochs_str = [str(e) for e in vis_epochs]

        
        
        title_prefix = f"With Gradient Link ({mode.capitalize()} Mode)" if with_gradient_link else "Without Gradient Link"
        visualize_attention(write_weights_list[0], f"{title_prefix}: Write Attention (Epoch {vis_epochs_str[0]})")
        visualize_attention(read_weights_list[0], f"{title_prefix}: Read Attention (Epoch {vis_epochs_str[0]})")
        visualize_attention(write_weights_list[1], f"{title_prefix}: Write Attention (Epoch {vis_epochs_str[1]})")
        visualize_attention(read_weights_list[1], f"{title_prefix}: Read Attention (Epoch {vis_epochs_str[1]})")

    
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(f"Loss History ({title_prefix})", fontsize=16)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print("Training finished.")



print("Running training for Without Gradient Link model...")
train_and_visualize(with_gradient_link=False, mode='overwrite')

print("\nRunning training for With Gradient Link (Overwrite Mode)...")
train_and_visualize(with_gradient_link=True, mode='overwrite')

print("\nRunning training for With Gradient Link (Add Mode)...")
train_and_visualize(with_gradient_link=True, mode='add')
