import argparse
import json

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--BATCH_SIZE', type=int,default=64)
parser.add_argument('--FILL_NAN', type=int,default=10)
parser.add_argument('--TIME_STEP', type=int, default=10)
parser.add_argument('--COLUMN', type=str, default='cpu_consumption')
parser.add_argument('--EPOCHS', type=int, default=2000)
parser.add_argument('--lr', type=float,default=0.001)
parser.add_argument('--HIDDEN_LAYER_SIZE', type=int, default=10)
parser.add_argument('--NUM_LAYERS', type=int,default=3)
parser.add_argument('--TRAIN_TEST_SPLOT', type=float,default=0.9)

# Parse the arguments
hyperparameters = parser.parse_args()

# Save the arguments to a dictionary
hyperparameters_dict = vars(hyperparameters)

# Save the dictionary to a JSON file
with open('hyperparameters.json', 'w') as f:
    json.dump(hyperparameters_dict, f)