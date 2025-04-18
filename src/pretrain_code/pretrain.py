import torch
import codecs

# Importing necessary modules
from ReGCN import ReGCNs
from torch_data import outputCSVfile
from dataprocessing import data_pre
from param import parameter_parser

# Set device to CUDA if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to write a list to a CSV file
def list_write_csv(file_name, l):
    with codecs.open(file_name, 'w') as f:  # Using 'with' to automatically close the file
        for line in l:
            f.write(str(line) + '\n')  # Write each item in the list to the file

# Pretraining function
def pretrain(model, train_data, optimizer, opt):
    epoch_list = []  # To store epoch numbers
    loss_list = []   # To store loss values for each epoch

    # Loop through the number of epochs
    for epoch in range(0, opt.epoch):
        print('epoch', epoch)
        model.zero_grad()  # Clear previous gradients
        feature, matrix = model(train_data)  # Get model outputs
        loss_fn = torch.nn.MSELoss(reduction='mean')  # Mean Squared Error loss function

        # Choose the loss functions based on the task type
        if opt.name == 'drug':
            # Loss calculation for drug-related tasks
            loss1 = loss_fn(matrix, train_data['dd_dr']['data_matrix'].to(device))
            loss2 = loss_fn(matrix, train_data['dd_di']['data_matrix'].to(device))
            loss3 = loss_fn(matrix, train_data['dd_se']['data_matrix'].to(device))
            loss4 = loss_fn(matrix, train_data['dd_str']['data_matrix'].to(device))
            loss5 = loss_fn(matrix, train_data['dd_pro']['data_matrix'].to(device))
            loss6 = loss_fn(matrix, train_data['dd_gs']['data_matrix'].to(device))
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        else:
            # Loss calculation for protein-related tasks
            loss1 = loss_fn(matrix, train_data['pp_pro']['data_matrix'].to(device))
            loss2 = loss_fn(matrix, train_data['pp_di']['data_matrix'].to(device))
            loss3 = loss_fn(matrix, train_data['pp_seq']['data_matrix'].to(device))
            loss4 = loss_fn(matrix, train_data['pp_dr']['data_matrix'].to(device))
            loss5 = loss_fn(matrix, train_data['pp_gs']['data_matrix'].to(device))
            loss = loss1 + loss2 + loss3 + loss4 + loss5

        epoch_list.append(epoch)  # Store the current epoch number
        loss_list.append(loss.item())  # Store the current loss value

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

        # Print loss and feature matrix for debugging
        print(epoch, loss.item())
        print(feature)
        print(matrix)

    # Normalize the matrix between 0 and 1
    scoremin, scoremax = matrix.min(), matrix.max()
    matrix = (matrix - scoremin) / (scoremax - scoremin)

    # Convert feature and matrix to numpy for further analysis
    feature = feature.cpu().detach().numpy()
    matrix = matrix.cpu().detach().numpy()

    # Save the results to CSV files using relative paths
    output_path = '../../dataset/LuoDTI/input/dim/'  # Relative path to the output directory
    if opt.name == 'drug':
        outputCSVfile(f'{output_path}{opt.name}_f{opt.f}_feature{opt.d_out_channels}.csv', feature)
        outputCSVfile(f'{output_path}{opt.name}_f{opt.f}_matrix{opt.d_out_channels}.csv', matrix)
        list_write_csv(f'{output_path}{opt.name}_f{opt.f}_recon{opt.d_out_channels}_losslist.csv', loss_list)
    else:
        outputCSVfile(f'{output_path}{opt.name}_f{opt.f}_feature{opt.p_out_channels}.csv', feature)
        outputCSVfile(f'{output_path}{opt.name}_f{opt.f}_matrix{opt.p_out_channels}.csv', matrix)
        list_write_csv(f'{output_path}{opt.name}_f{opt.f}_recon{opt.p_out_channels}_losslist.csv', loss_list)

    return feature, matrix  # Return the feature and matrix for further use


# Main function to set up the model, data, and training process
def main(name):
    args = parameter_parser(name)  # Parse command-line arguments
    pretrain_data = data_pre(args)  # Preprocess the data
    torch.cuda.empty_cache()  # Clear the GPU memory cache to avoid memory issues

    model = ReGCNs(args)  # Initialize the model
    model.to(device)  # Move the model to the appropriate device (GPU/CPU)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)  # Set up the optimizer
    feature, matrix = pretrain(model, pretrain_data, optimizer, args)  # Pretrain the model

# Entry point of the script
if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear the GPU memory cache at the start
    main("drug")  # Run the pretraining for the "drug" task
    main("protein")  # Run the pretraining for the "protein" task
