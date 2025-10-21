import logging
import torch
from src.dataset import load_dataset
from src.model import BaseModel

def main():
    logging.basicConfig(
        level=logging.INFO, 
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    ## Prepare datasets
    logging.info("Prepare datasets...")
    dataset, _, _, _, class_weight_dict = load_dataset()
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    ## Set up device
    logging.info(f"XPU is available: {torch.xpu.is_available()}")
    device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    ## Train the model on the training set and evaluate on test set
    model = BaseModel(device=device)
    model.fit_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_weight_dict=class_weight_dict
    )
    # Export to HF Hub
    model.save_model()



if __name__ == "__main__":
    main()
