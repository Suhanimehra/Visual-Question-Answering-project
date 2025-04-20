import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class CustomVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        # Image encoder (CNN)
        self.cnn = nn.Sequential(
            *list(torch.hub.load('pytorch/vision', 'resnet18', pretrained=True).children())[:-1]
        )
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 1024),  # ResNet18:512, BERT:768
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_answers)
        )
    
    def forward(self, images, questions):
        # Image features
        img_features = self.cnn(images).squeeze()
        
        # Text features
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
        text_features = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        
        # Combine
        combined = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined)