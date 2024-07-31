import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_layer_size,
                 num_layers , 
                 output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers , batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
     
        return predictions

     
    def plot_results(self,dataset,dataloader,time_step,device,save_folder=None):
        results = torch.zeros(time_step,1).to(device)

        for data,_ in dataloader:
            res  = self(data)
            results = torch.cat((results,res),dim=0)
        plt.plot(results.detach().cpu(),color='r',label='Predicted')
        plt.plot(dataset.df.values,color='b',label='Real')
        plt.legend()
        
        plt.tight_layout()
        if save_folder is  None:
            plt.show()
        else:
            test_file= save_folder+'/test.png'
            plt.savefig(test_file)
        plt.clf()


def train_lstm(model,epochs,train_dataloader,lr,criterion=None,optimizer=None,save_folder=None):
    if criterion is None:
        criterion = nn.MSELoss()  # or any other loss function based on your task
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)  # or any other optimizer

    
    losses = []
    # Training loop
    for epoch in range(epochs):  # number of epochs
        accumulative_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accumulative_loss += loss.item()
        losses.append(accumulative_loss)
        if epoch %10 ==0:
            print(f"Epoch {epoch}, loss: {accumulative_loss / len(train_dataloader)}")


    print('Finished Training')
    plt.plot(losses)
    if save_folder is  None:
        plt.show()
    else:
        loss_file=  save_folder+'/loss.png'
        plt.savefig(loss_file)
        model_file = save_folder + '/model.pt'
        torch.save(model.state_dict(), model_file)
    plt.clf()
    
   