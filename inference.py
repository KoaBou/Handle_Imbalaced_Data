import cv2

from data import DataModule
from model import ResNetModel
import torch
import torchmetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor:
    def __init__(self, model_path):
        self.mode_path = model_path

        self.model = ResNetModel.load_from_checkpoint(model_path)
        self.model.to(DEVICE).eval().freeze()

        self.proccessor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = self.proccessor.label_to_age

    def preprocessing(self, images):
        images = self.proccessor.transform(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return images

    def predict(self, images):
        images = self.preprocessing(images)
        logits = self.model(images)
        scores = self.softmax(logits[0]).tolist()
        predictions = []

        for score, label in zip(scores, self.labels):
            predictions.append(({"label": label, "score":score}))

        return predictions

if __name__ == "__main__":
    predictor = Predictor("models/epoch=9-step=400.ckpt")
    file_path = "dataset/custom_korean_family_dataset_resolution_128/test_images/F0853_AGE_F_41_d1.jpg"
    image = cv2.imread(file_path)
    print(predictor.predict(image))

