import os 
import pandas  as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters
hyperparameters = read_json_file('hyperparameters.json')



BATCH_SIZE = hyperparameters['BATCH_SIZE']
FILL_NAN= hyperparameters['FILL_NAN']
TIME_STEP =hyperparameters['TIME_STEP']
COLUMN = hyperparameters['COLUMN'] # 'cpu_consumption
EPOCHS = hyperparameters['EPOCHS']
lr  = hyperparameters['lr'] 
HIDDEN_LAYER_SIZE =     hyperparameters['HIDDEN_LAYER_SIZE']
NUM_LAYERS= hyperparameters['NUM_LAYERS']
TRAIN_TEST_SPLOT=   hyperparameters['TRAIN_TEST_SPLOT']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_folder='lstm/logs'


def train_test_split(df,split=TRAIN_TEST_SPLOT):
    train_split = int(len(df)*split)
    test_split=  len(df) -train_split
    return train_split,test_split

class LoadPredictionDataset(Dataset):
    def __init__(self,df,
                 start_index,
                 population,                 
                 time_step=TIME_STEP,
                 column=COLUMN,
                 device=device):
        previous_overflow=  max(start_index-time_step,0)
        df = df[column]
        self.df = df.iloc[previous_overflow:start_index+population]
        self.column = column
        self.time_step = time_step
        self.device = device
        self.length = len(self.df)-self.time_step -1
        
    def __getitem__(self, index):
        previous_values = self.df.iloc[index:index+self.time_step].values
        previous_values = torch.tensor(previous_values).unsqueeze(0)
        previous_values = previous_values.float().to(self.device)
        target_values = self.df.iloc[index+self.time_step]
        target_values = torch.tensor(target_values).float().to(self.device)
        target_values = target_values.unsqueeze(0)
        return previous_values, target_values
    
    def __len__(self):
        return self.length
        

class LSTMModel(nn.Module):
    def __init__(self, input_size=TIME_STEP, hidden_layer_size=HIDDEN_LAYER_SIZE,num_layers =NUM_LAYERS, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers , batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)



    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
     
        return predictions

def plot_results(model,dataset,dataloader,column=COLUMN,device=device,save_folder=None):
    results = torch.zeros(TIME_STEP,1).to(device)

    for data,_ in dataloader:
        res  = model(data)
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

def train(model,train_dataloader,device=device,criterion=None,optimizer=None,save_folder=None):
    if criterion is None:
        criterion = nn.MSELoss()  # or any other loss function based on your task
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)  # or any other optimizer

    
    losses = []
    # Training loop
    for epoch in range(EPOCHS):  # number of epochs
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


os.makedirs(log_folder,exist_ok=True)
index_filepath  =log_folder+'/index.txt'
if not os.path.exists(index_filepath):
    with open(index_filepath, 'w') as file:
        file.write('0')
        run_index = 0
else:
    with open(index_filepath, 'r') as file:
        run_index = int(file.read().strip())
    run_index += 1
    with open(index_filepath, 'w') as file:
        file.write(str(run_index))
    
run_folder = log_folder + '/run_' + str(run_index)
os.makedirs(run_folder)


csv_file=  'lstm/results_cpu_memory_eco-efficiency.csv'
df = pd.read_csv(csv_file)
df_ffill = df.ffill(limit=FILL_NAN)
df_bfill = df.bfill(limit=FILL_NAN)
df = (df_ffill + df_bfill) / 2

# Initialize a new StandardScaler instance
scaler = StandardScaler()

# Fit and transform the DataFrame
df  = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


train_split,test_split = train_test_split(df)
train_dataset = LoadPredictionDataset(df,start_index=0,population=train_split,device=device)
test_dataset  = LoadPredictionDataset(df,start_index=train_split,population=test_split,device=device)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel().to(device)


train(model,train_dataloader,save_folder=run_folder,device=device)
plot_results(model,test_dataset,test_dataloader,device=device,save_folder=run_folder)


# Save the dictionary to a JSON file
with open(run_folder+'/hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)