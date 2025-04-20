import os
import sys
import json
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import your custom modules
from models.custom_cnn import CustomVQAModel
from src.data_loader import VQADataset

def load_config(config_path="configs/train.yaml"):
    """Load training configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def train():
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load answer vocabulary
    vocab_path = "data/processed/vocab.json"
    with open(vocab_path) as f:
        answer_vocab = json.load(f)
    
    # Initialize model
    model = CustomVQAModel(len(answer_vocab)).to(device)
    print(f"Model initialized with {len(answer_vocab)} answer classes")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # Create datasets
    train_dataset = VQADataset(
        annotations_path=os.path.join("data", "vqa_v2", "Annotations", "v2_mscoco_train2014_annotations.json"),
        questions_path=os.path.join("data", "vqa_v2", "Questions", "v2_OpenEnded_mscoco_train2014_questions.json"),
        images_dir=os.path.join("data", "vqa_v2", "Images", "train2014")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0
    )

    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, questions, answers) in enumerate(train_loader):
            images, answers = images.to(device), answers.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")

        # Save checkpoint
        os.makedirs("models/saved_models", exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join("models", "saved_models", f"epoch_{epoch+1}.pth")
        )

if __name__ == "__main__":
    train()