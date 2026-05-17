from models.adapter import FeatureAdapter
import os
import torch
import numpy as np

class MedicalAnalyzer:
    def __init__(self, encoder, device="cpu"):
        self.encoder = encoder
        self.device = device
        
        # Initialize and load Adapter if exists
        self.adapter = FeatureAdapter().to(device)
        ckpt_path = "../checkpoints/medical_adapter.pth"
        self.has_adapter = False
        if os.path.exists(ckpt_path):
            try:
                self.adapter.load_state_dict(torch.load(ckpt_path, map_location=device))
                self.adapter.eval()
                self.has_adapter = True
                print("Medical Feature Adapter loaded successfully.")
            except:
                print("Failed to load Medical Adapter. Using base embeddings.")

        # Validation categories
        self.validation_categories = [
            "chest xray", "MRI scan", "CT scan", "pathology slide", 
            "ultrasound image", "endoscopy frame", "selfie", 
            "animal photo", "document screenshot"
        ]
        self.validation_tokens = self.encoder.extract_text_features(self.validation_categories)

        # Modalities
        self.modalities = [
            "Chest X-ray", "MRI Scan", "CT Scan", 
            "Pathology Slide", "Ultrasound", "Endoscopy"
        ]
        self.modality_tokens = self.encoder.extract_text_features(self.modalities)

    def refine_features(self, image_features):
        if self.has_adapter:
            with torch.no_grad():
                return self.adapter(image_features.float())
        return image_features

    def validate_upload(self, image_features):
        refined = self.refine_features(image_features)
        probs = self.encoder.get_similarity(refined, self.validation_tokens)
        probs = probs.cpu().numpy()[0]
        
        medical_indices = [0, 1, 2, 3, 4, 5]
        non_medical_indices = [6, 7, 8]
        medical_score = np.sum(probs[medical_indices])
        non_medical_score = np.sum(probs[non_medical_indices])
        
        return {
            "is_valid": bool(medical_score > non_medical_score),
            "medical_confidence": float(medical_score),
            "top_category": self.validation_categories[np.argmax(probs)]
        }

    def detect_modality(self, image_features):
        refined = self.refine_features(image_features)
        probs = self.encoder.get_similarity(refined, self.modality_tokens)
        probs = probs.cpu().numpy()[0]
        idx = np.argmax(probs)
        return {
            "modality": self.modalities[idx],
            "confidence": float(probs[idx])
        }

    def predict_disease(self, image_features, modality):
        queries = self.get_disease_queries(modality)
        text_features = self.encoder.extract_text_features(queries)
        
        refined = self.refine_features(image_features)
        probs = self.encoder.get_similarity(refined, text_features).cpu().numpy()[0]
        
        idx = np.argmax(probs)
        
        # Clinical Bias: 
        # For Endoscopy, we use a higher threshold (0.35) to avoid over-flagging 
        # reflections and artifacts as diseases.
        threshold = 0.35 if modality == "Endoscopy" else 0.25
        
        if idx == 0 and len(probs) > 1:
            abnormal_probs = probs[1:]
            if np.max(abnormal_probs) > threshold:
                idx = np.argmax(abnormal_probs) + 1
        
        prediction = queries[idx].split(',')[0].strip()
        # Clean up the long descriptive names for the UI
        display_map = {
            "normal healthy gastrointestinal mucosa": "Normal Mucosa",
            "gastric polyp or sessile lesion": "Polyp/Lesion",
            "active GI ulcer or ulcerative colitis": "Ulcer/Colitis",
            "esophagitis with visible inflammation": "Esophagitis",
            "normal cecum with appendiceal orifice": "Normal Cecum",
            "normal pylorus or stomach outlet": "Normal Pylorus",
            "normal z-line junction": "Normal Z-Line",
            "barretts esophagus with specialized epithelium": "Barrett's Esophagus"
        }
        prediction = display_map.get(prediction, prediction)
        
        confidence = float(probs[idx])
        return prediction, confidence

    def get_disease_queries(self, modality):
        queries = {
            "Chest X-ray": ["Normal lungs", "Pneumonia, opacity", "Effusion", "Cardiomegaly"],
            "MRI Scan": ["Normal brain", "Brain tumor, mass", "Stroke", "Sclerosis"],
            "CT Scan": ["Normal abdomen", "Hemorrhage", "Inflammation", "Fracture"],
            "Pathology Slide": [
                "normal healthy tissue architecture in a pathology slide", 
                "malignant cancerous tissue with architectural distortion", 
                "acute inflammation and leukocyte infiltration in tissue", 
                "necrotic cell debris and tissue death"
            ],
            "Ultrasound": ["Normal scan", "Gallstones", "Cystic lesion"],
            "Endoscopy": [
                "normal healthy gastrointestinal mucosa", 
                "gastric polyp or sessile lesion", 
                "active GI ulcer or ulcerative colitis", 
                "esophagitis with visible inflammation",
                "normal cecum with appendiceal orifice",
                "normal pylorus or stomach outlet",
                "normal z-line junction",
                "barretts esophagus with specialized epithelium"
            ]
        }
        return queries.get(modality, ["Normal", "Abnormal"])
