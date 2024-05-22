import os
import json
class Bookkeeper():
    def __init__(self,
                 log_folder='logs'):
        
        self.log_folder = log_folder
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
            
        self.run_folder = log_folder + '/run_' + str(run_index)
        os.makedirs(self.run_folder)
    
        self.hyperparameters = self.read_json_file('hyperparameters.json')
                
        with open(self.run_folder+'/hyperparameters.json', 'w') as f:
            json.dump(self.hyperparameters, f)
            
    def get_run_folder(self):
        return self.run_folder  
    
    def get_hyperparameters(self):
        return self.hyperparameters
        
        
    def read_json_file(self,file_path):
        with open(file_path, 'r') as f:
            hyperparameters = json.load(f)
        return hyperparameters
