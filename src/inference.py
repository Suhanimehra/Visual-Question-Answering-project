import torch
from PIL import Image
from models.custom_cnn import CustomVQAModel
from data_loader import VQADataset

def load_model(model_path, vocab_path):
    model = CustomVQAModel(len(vocab))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def answer_question(image_path, question, model, vocab):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image, [question])
        pred = output.argmax(-1).item()
    
    return {v: k for k, v in vocab.items()}[pred]

if __name__ == "__main__":
    # Example usage
    vocab = json.load(open("data/processed/vocab.json"))
    model = load_model("models/saved_models/best_model.pth", vocab)
    
    result = answer_question(
        image_path="data/sample.jpg",
        question="What color is the car?",
        model=model,
        vocab=vocab
    )
    print("Answer:", result)