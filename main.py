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
    train_dataset, validation_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

    ## Set up device
    logging.info(f"XPU is available: {torch.xpu.is_available()}")
    device = "auto" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    ## Train the model on the training set and evaluate on test set
    model = BaseModel(
        model_name="minishlab/potion-base-32m",
        device=device
    )

    model.fit_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=15,
        class_weight_dict=class_weight_dict
    )
    # Export to HF Hub
    model.save_model(model_name="ticket_triage_potion-base-32M_1e-3")



if __name__ == "__main__":
    main()
