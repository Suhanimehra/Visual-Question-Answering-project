import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VQADataset(Dataset):
    def __init__(self, annotations_path, questions_path, images_dir, transform=None, 
                 answer_vocab_path="data/processed/vocab.json", min_occurrences=5):
        """
        Args:
            annotations_path: Path to VQA annotations JSON
            questions_path: Path to VQA questions JSON
            images_dir: Directory containing COCO images
            transform: Optional image transforms
            answer_vocab_path: Where to save/load answer vocabulary
            min_occurrences: Minimum times an answer must appear to be included
        """
        # Convert to absolute paths
        self.images_dir = os.path.abspath(images_dir)
        annotations_path = os.path.abspath(annotations_path)
        questions_path = os.path.abspath(questions_path)
        
        # Create processed directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(answer_vocab_path)), exist_ok=True)
        
        # Default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # This converts PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load questions and annotations
        with open(questions_path) as f:
            self.questions = json.load(f)['questions']
        with open(annotations_path) as f:
            self.annotations = json.load(f)['annotations']
        
        # Track missing images
        self.missing_images = 0
        
        # Create or load answer vocabulary
        if os.path.exists(answer_vocab_path):
            with open(answer_vocab_path) as f:
                self.answer_vocab = json.load(f)
        else:
            self.answer_vocab = self._create_answer_vocab(answer_vocab_path, min_occurrences)
    
    def _create_answer_vocab(self, save_path, min_occurrences):
        answer_counts = {}
        for ann in self.annotations:
            for answer in ann['answers']:
                answer_counts[answer['answer']] = answer_counts.get(answer['answer'], 0) + 1
        
        answer_vocab = {
            ans: idx for idx, (ans, cnt) in enumerate(
                sorted(answer_counts.items(), 
                     key=lambda x: x[1], 
                     reverse=True))
            if cnt >= min_occurrences
        }
        answer_vocab['<UNK>'] = len(answer_vocab)
        
        with open(save_path, 'w') as f:
            json.dump(answer_vocab, f, indent=2)
        return answer_vocab
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]['question']
        image_id = self.questions[idx]['image_id']
        
        # Construct image path
        img_name = f"COCO_train2014_{image_id:012d}.jpg"
        img_path = os.path.join(self.images_dir, img_name)
        
        # Initialize blank image tensor (will be returned if image is missing)
        blank_image = torch.zeros(3, 224, 224)  # Direct tensor creation
        
        # Handle missing images
        if not os.path.exists(img_path):
            self.missing_images += 1
            if self.missing_images <= 10:
                print(f"Warning: Missing image {img_path}")
            elif self.missing_images == 11:
                print("Additional missing images not shown...")
            return blank_image, question, self.answer_vocab['<UNK>']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)  # transform converts to tensor
            else:
                # Convert to tensor if no transform provided
                image = transforms.ToTensor()(image)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return blank_image, question, self.answer_vocab['<UNK>']
        
        # Get most frequent answer
        answers = [a['answer'] for a in self.annotations[idx]['answers']]
        answer = max(set(answers), key=answers.count)
        answer_idx = self.answer_vocab.get(answer, self.answer_vocab['<UNK>'])
        
        return image, question, answer_idx

    def get_missing_count(self):
        """Returns number of missing images encountered"""
        return self.missing_images