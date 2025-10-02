import os
import matplotlib.pyplot as plt

def save_history(save_dir, loss, trained_data):
    """
    Plot and save the diffusion and diffusion history loaded in the diffusion object

    Function arguments:
        save_dir (str) - Directory to save the history files
        loss (list) - List of loss values to plot
        trained_data (list) - List of trained data to save

    Returns:
        None
    """
    # Make directories that don't exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the current diffusorerate loss progress
    with open(f"{save_dir}diffusor_loss.txt", "w") as file:
        for logged_loss in loss:
            file.write(f"{logged_loss}\n")


    with open(f"{save_dir}trained.txt", "w") as file:
        for trained in trained_data:
            file.write(f"{trained}\n")

def load_history(save_dir):
    """
    Load diffusion/diffusion loss history and trained data from text files
    and assign them to the diffusion object.

    Function arguments:
        save_dir (str) - Directory where the history files are stored

    Returns:
        loss (list) - List of loss values loaded from the file
        trained_data (list) - List of trained data loaded from the file
    """
    import os

    diffusor_path = os.path.join(save_dir, "diffusor_loss.txt")
    trained_path = os.path.join(save_dir, "trained.txt")

    # Initialize containers
    loss = []
    trained_data = []

    # Load diffusion loss
    if os.path.exists(diffusor_path):
        with open(diffusor_path, "r") as file:
            loss = [float(line.strip()) for line in file if line.strip()]
    
    # Load trained data
    if os.path.exists(trained_path):
        with open(trained_path, "r") as file:
            trained_data = [line.strip() for line in file if line.strip()]
    
    return loss, trained_data

