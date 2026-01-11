import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def define_data_set(num_tasks, T):

    x_dataset = torch.zeros([num_tasks, 4, T]); y_dataset = []; 
    
    #coin flips for ground truth
    context_ = 2*np.random.rand(num_tasks) - 1           #-1 = motion context; +1 = color context
    context = 2*((context_<0.0).astype(int))-1      #-1 = motion context; +1 = color context
    
    color_ = 2*np.random.rand(num_tasks) - 1             #-1 = blue; +1 = red
    color = 2*((color_<0.0).astype(int)) - 1               #-1 = blue; +1 = red
    
    motion_ = 2*np.random.rand(num_tasks) - 1 #0 = left; 1 = right
    motion = 2*((motion_<0.0).astype(int)) - 1 #0 = left; 1 = right

    y_dataset_full = torch.cat((torch.tensor(motion).unsqueeze(1), torch.tensor(color).unsqueeze(1), torch.tensor(context).unsqueeze(1)), dim=1)

    for n in range(num_tasks):

        if context[n] == 0:
            y_dataset.append(motion[n])        
        else:
            y_dataset.append(color[n])


        motion_signal = motion_[n] + motion_[n]/10*np.random.randn(T)        
        color_signal = color_[n] + color_[n]/10*np.random.randn(T)        

        motion_signal = 0.2*motion_signal; color_signal = 0.2*color_signal;
        
        x_dataset[n,0,:] = torch.tensor(motion_signal)
        x_dataset[n,1,:] = torch.tensor(color_signal)
        if context[n] == 0:
            x_dataset[n,2,:] = torch.tensor([1]*T)
            x_dataset[n,3,:] = torch.tensor([0]*T)
        else:
            x_dataset[n,2,:] = torch.tensor([0]*T)
            x_dataset[n,3,:] = torch.tensor([1]*T)
            
    y_dataset = torch.tensor(y_dataset)

    return x_dataset, y_dataset, y_dataset_full


x_dataset, y_dataset, y_dataset_full = define_data_set(10000, 750)
x_dataset.shape, y_dataset.shape

training_dataset = TensorDataset(x_dataset, y_dataset)
training_data = DataLoader(training_dataset, batch_size=50, shuffle=True)

class RNN_neuro(nn.Module):
    
     def __init__(self, input_size=4, hidden_size=100, output_size = 1, batchsize = 50, dt=1, **kwargs):
        super().__init__()
         
        self.J_weights = nn.Parameter(1.5*torch.randn(hidden_size, hidden_size)/np.sqrt(hidden_size), requires_grad=True)
        self.nonlinear_fn = nn.Tanh()
    
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.bias = nn.Parameter(torch.randn(hidden_size), requires_grad = True)

        self.std = 0.1
         
        self.tau = 10;
        self.dt = 1
        self.T = 750

        self.batchsize = batchsize
        self.hidden_size = hidden_size
         
     def forward(self, input_):

        input_ = input_.float()
         
        x = torch.zeros(self.batchsize, self.hidden_size, device=input_.device)
        r = self.nonlinear_fn(x)

        all_x = [x]; all_r = [r]; 
        all_z = [self.fc_out(r)];

        for t in range(self.T):
            temp = self.fc_in(input_[:,:,t]) + self.bias
            noise = self.std*torch.randn(self.batchsize, self.hidden_size) 

            x = x + self.dt/self.tau * (-x + r @ self.J_weights.T + temp + noise)
            r = self.nonlinear_fn(x)
            out = self.fc_out(r)

            all_x.append(x); all_r.append(r); 
            all_z.append(out)

        all_x = torch.stack(all_x, dim=2)
        all_r = torch.stack(all_r, dim=2)
        all_z = torch.stack(all_z, dim=2)

        return all_z, all_x, all_r

model = RNN_neuro()
model.float()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()   # Mante uses MSE on final output

def train_rnn(model, training_data, num_epochs=50, batchsize=50, lr=1e-3):

    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_acc = 0
        for bid, batch in enumerate(training_data):

            X, y = batch
            batchsize = X.shape[0]

            # forward pass
            all_z, all_x, all_r = model(X)

            # final output at last time step, shape (batch, 1)
            z_final = all_z[:, :, -1].squeeze(-1)

            optimizer.zero_grad()
            # MSE loss target needs shape (batch,)
            loss = loss_fn(z_final, y.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            preds = torch.sign(z_final)
            correct = (preds == y).sum().item()
            epoch_acc += correct
            epoch_total += batchsize
            
            print("z_final", z_final)

        epoch_acc = epoch_acc / epoch_total
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

    return epoch_loss, epoch_acc

epoch_loss, epoch_acc = train_rnn(model, training_data)