import os, logging
import datasets as ds
from model2vec.train import StaticModelForClassification
from model2vec.inference import StaticModelPipeline
from dotenv import load_dotenv

class BaseModel:
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-32m",
        device: str = "xpu"
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
        test_dataset: ds.Dataset,
        class_weight_dict: dict
    ):
        # Create X and y
        X_train, y_train = train_dataset["text"], train_dataset["labels"]
        X_test, y_test  = test_dataset["text"], test_dataset["labels"]


        # Train the classifier
        self.trained_classifier: StaticModelForClassification = self.classifier.fit(
            X=X_train, 
            Y=y_train,
            learning_rate=1e-4,
            batch_size=32,
            max_epochs=15,
            early_stopping_patience=3,
            device=self.device,
            class_weight=list(class_weight_dict.values()))

        # Evaluate the classifier
        results = self.trained_classifier.evaluate(X_test, y_test)
        logging.info(results)
    
    def save_model(
        self
    ):
        load_dotenv()
        HF_USER = os.getenv("HF_USER")
        HF_TOKEN = os.getenv("HF_TOKEN")
        pipeline = self.trained_classifier.to_pipeline()
        pipeline.push_to_hub(f"{HF_USER}/model2vec-ticket-triage", token=HF_TOKEN)
    
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