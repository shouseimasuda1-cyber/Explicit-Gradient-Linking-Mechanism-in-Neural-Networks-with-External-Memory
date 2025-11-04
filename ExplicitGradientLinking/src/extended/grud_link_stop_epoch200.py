import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
grad_list = {}


def read_with_attention(memory_tensor, query, id_token, use_grad_link):
    B, N, D = memory_tensor.shape
    attn_logits = torch.bmm(memory_tensor, query.unsqueeze(2)).squeeze(2)
    attn_weights = F.softmax(attn_logits, dim=1)
    read_output = torch.bmm(attn_weights.unsqueeze(1), memory_tensor).squeeze(1)
    if use_grad_link:
        def save_grad_hook(grad):
            grad_list[id_token] = grad

        read_output.register_hook(save_grad_hook)
    return read_output, attn_weights


class WriteFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, id_token, memory_tensor, write_weights, use_grad_link):
        ctx.save_for_backward(input_tensor, write_weights, memory_tensor)
        ctx.id_token = id_token
        ctx.use_grad_link = use_grad_link

        memory_out = memory_tensor.clone()
        write_weights_expanded = write_weights.unsqueeze(2) 
        input_tensor_expanded = input_tensor.unsqueeze(1)   
        memory_out = memory_out * (1 - write_weights_expanded) + input_tensor_expanded * write_weights_expanded
        return memory_out

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, write_weights, memory_tensor = ctx.saved_tensors
        id_token = ctx.id_token
        use_grad_link = ctx.use_grad_link

        
        grad_memory = grad_output.clone()  

        
        
        
        if use_grad_link:
             grad_read = grad_list.pop(id_token, None)
             if grad_read is not None:
                 grad_memory = grad_memory + grad_read.detach().unsqueeze(1)  


        
        
        grad_input = torch.sum(write_weights.unsqueeze(2) * grad_memory, dim=1)  

        
        grad_weights = torch.sum(
            grad_memory * (input_tensor.unsqueeze(1) - memory_tensor),
            dim=2
        )  

        
        grad_memory_final = grad_memory * (1 - write_weights.unsqueeze(2))  

        
        return grad_input, None, grad_memory_final, grad_weights, None

def write_to_memory(input_tensor, id_token, memory_tensor, write_weights, use_grad_link):
    return WriteFunction.apply(input_tensor, id_token, memory_tensor, write_weights, use_grad_link)


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

    def write_batch(self, x_batch, t, use_grad_link):
        mem_values = self.write_net(x_batch)  

        
        
        write_query = self.write_query_net(x_batch) 
        
        write_logits = torch.bmm(self.memory, write_query.unsqueeze(2)).squeeze(2)
        write_weights = F.softmax(write_logits, dim=1)  

        
        
        id_token = f"{t}"
        self.memory = write_to_memory(mem_values, id_token, self.memory, write_weights, use_grad_link)
        return self.memory

    def read_batch(self, x_batch, t, use_grad_link):
        query = self.read_query_net(x_batch)
        
        
        id_token = f"{t}"
        read_result, _ = read_with_attention(self.memory, query, id_token, use_grad_link)
        return self.read_out_net(read_result)


def train_model(epochs, stop_grad_link_epoch=None, model=None):
    grad_list.clear()

    input_dim = 80
    mem_dim = 150
    num_slots = 100
    seq_len = 100
    batch_size = 32
    lr = 1e-3
    clip_value = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if model is None:
        model = AttentionMemoryInterface(input_dim, mem_dim, num_slots).to(device)
    else:
        model.to(device) 


    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_list = []
    total_grad_norm_list = []

    for epoch in range(epochs):
        use_grad_link = True if stop_grad_link_epoch is None else epoch < stop_grad_link_epoch

        grad_list.clear()

        x_seq = torch.randn(batch_size, seq_len, input_dim).to(device)
        model.reset_memory(batch_size, device=device)

        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            model.write_batch(x_t, t, use_grad_link)

        outputs = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            out = model.read_batch(x_t, t, use_grad_link)
            outputs.append(out.unsqueeze(1))

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

        optimizer.step()

        loss_list.append(loss.item())
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Total Grad Norm = {total_norm:.6f}, Grad Link Active: {use_grad_link}")

    return loss_list, total_grad_norm_list, model 


epochs = 600
stop_epoch = 200


input_dim = 80
mem_dim = 160
num_slots = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_model = AttentionMemoryInterface(input_dim, mem_dim, num_slots).to(device)


initial_state_dict = copy.deepcopy(initial_model.state_dict())


print(f"--- Training with Gradient Link stopping at epoch {stop_epoch} ---")

model_stop_link = AttentionMemoryInterface(input_dim, mem_dim, num_slots).to(device)
model_stop_link.load_state_dict(copy.deepcopy(initial_state_dict))
loss_stop_link, grad_norm_stop_link, _ = train_model(epochs, stop_grad_link_epoch=stop_epoch, model=model_stop_link)


print("\n--- Training with Gradient Link always active ---")

model_always_link = AttentionMemoryInterface(input_dim, mem_dim, num_slots).to(device)
model_always_link.load_state_dict(copy.deepcopy(initial_state_dict))
loss_always_link, grad_norm_always_link, _ = train_model(epochs, stop_grad_link_epoch=None, model=model_always_link)



plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(loss_stop_link, label=f"Link stops at Epoch {stop_epoch}")
plt.plot(loss_always_link, label="Link always active")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Copy Task Loss Comparison")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(grad_norm_stop_link, label=f"Link stops at Epoch {stop_epoch}")
plt.plot(grad_norm_always_link, label="Link always active")
plt.xlabel("Epoch")
plt.ylabel("Total Gradient Norm")
plt.title("Total Gradient Norm Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
