import os, logging
import datasets as ds
import torch
from model2vec.train import StaticModelForClassification
from model2vec.inference import StaticModelPipeline
from dotenv import load_dotenv
from src.dataset import oversample_with_interleave

class BaseModel:
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-32m",
        device: str = "auto"
    ):
        logging.basicConfig(
            level=logging.INFO, 
            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.model_name = model_name
        self.device = device
        self.classifier = StaticModelForClassification.from_pretrained(model_name=self.model_name)
    
    def fit_model(
        self,
        train_dataset : ds.Dataset,
        validation_dataset: ds.Dataset,
        test_dataset: ds.Dataset,
        learning_rate: float,
        max_epochs: int,
        batch_size: int,
        class_weight_dict: dict,
        oversample: bool,
        num_labels: int
    ):
        if oversample:
            ## Oversampling (underrepresented) classes
            logging.info("Oversampling (underrepresented) classes...")
            train_dataset = oversample_with_interleave(
                train_dataset,
                num_labels=num_labels,
                boost_classes=[2,3,7],
                boost_factor=3.5,
                seed=10
            )
            labels = sorted(train_dataset.to_pandas()["labels"].unique())
            class_weight_dict = {}
            for l in labels:
                class_weight_dict.update({l: train_dataset.to_pandas()["labels"].apply(lambda x: x == l).sum() / train_dataset.to_pandas()["labels"].count()})
                logging.info(f"[Train] Label {l}: {train_dataset.to_pandas()["labels"].apply(lambda x: x == l).sum()} occurrences")
        
        # Create X and y
        X_train, y_train = train_dataset["text"], train_dataset["labels"]
        X_val, y_val = validation_dataset["text"], validation_dataset["labels"]
        X_test, y_test  = test_dataset["text"], test_dataset["labels"]
    
        # Train the classifier
        self.trained_classifier: StaticModelForClassification = self.classifier.fit(
            X=X_train, 
            y=y_train,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=5,
            device=self.device,
            class_weight=torch.Tensor(list(class_weight_dict.values())),
            X_val=X_val,
            y_val=y_val
        )

        # Evaluate the classifier
        results = self.trained_classifier.evaluate(X_test, y_test)
        logging.info(f"\n{results}")
    
    def save_model(
        self,
        model_name
    ):
        load_dotenv()
        HF_USER = os.getenv("HF_USER")
        HF_TOKEN = os.getenv("HF_TOKEN")
        pipeline = self.trained_classifier.to_pipeline()
        pipeline.push_to_hub(f"{HF_USER}/{model_name}", token=HF_TOKEN, private=True)
    
class TicketTriageModel:
    def __init__(
        self,
        model_name :str  
    ):
        self.model_name = model_name
    
    def load_model(
        self,
    ):
        self.pipeline = StaticModelPipeline.from_pretrained(self.model_name)
        return self.pipeline