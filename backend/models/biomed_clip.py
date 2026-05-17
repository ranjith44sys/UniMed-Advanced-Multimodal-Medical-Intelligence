import torch
import open_clip
from PIL import Image
import os

class BioMedCLIPEncoder:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing BioMedCLIP on {self.device}...")
        
        # Load BioMedCLIP from Microsoft
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        self.model.to(self.device)
        self.model.eval()

    def extract_image_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            return image_features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            # Return a zero vector of correct dimension (512 for BioMedCLIP)
            return torch.zeros(1, 512).to(self.device)

    def extract_text_features(self, text_list):
        text_tokens = self.tokenizer(text_list).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def get_similarity(self, image_features, text_features):
        with torch.no_grad():
            logits_per_image = 100. * image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1)
        return probs
