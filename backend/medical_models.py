import torch
import torch.nn as nn
import clip
from PIL import Image
import os
import pickle
import numpy as np
import faiss

# Clinical Radiology Config
CANDIDATE_CAPTIONS = [
    "a chest x-ray scan",
    "a brain MRI study",
    "an abdominal CT scan",
    "pneumonia infection in the lungs",
    "healthy clear lungs",
    "normal brain tissue on MRI",
    "abnormal pathology on MRI",
    "normal abdominal organs on CT",
    "infected or abnormal CT results"
]

DETAILED_TRUTH = {
    "chest_normal": "Chest X-ray showing no acute cardiopulmonary abnormalities. Lungs are clear and cardiac silhouette is within normal limits.",
    "chest_pneumonia": "Chest X-ray showing consolidation and inflammatory opacities consistent with clinical pneumonia.",
    "mri_normal": "Magnetic Resonance Imaging showing normal anatomical brain structures with no signs of pathological infection.",
    "mri_infected": "Magnetic Resonance Imaging demonstrating tissue signal abnormalities consistent with clinical infection or inflammation.",
    "ct_normal": "Computed Tomography scan showing clear internal structures with no radiological evidence of disease.",
    "ct_infected": "Computed Tomography scan highlighting localized densities suggestive of an active infection site."
}

class CLIPAdaptor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super(CLIPAdaptor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        return self.layernorm(x + residual)

class UniMedModel:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading UniMedAI Brain on {self.device}...")
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("CLIP foundation loaded.")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            self.model = None

        self.adaptor = CLIPAdaptor().to(self.device)
        self.adaptor.eval()

        # FAISS Index and Metadata
        self.index = None
        self.metadata = None
        self.caption_embeddings = None
        self.load_faiss_assets()

    def load_faiss_assets(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, "index_data", "clip_index.faiss")
        metadata_path = os.path.join(base_dir, "index_data", "metadata.pkl")

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"FAISS Index Loaded: {self.index.ntotal} vectors")
        else:
            print("FAISS Index not found. Similarity search disabled.")

        if self.model:
            text_tokens = clip.tokenize(CANDIDATE_CAPTIONS).to(self.device)
            with torch.no_grad():
                ce = self.model.encode_text(text_tokens)
                self.caption_embeddings = (ce / ce.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()

    def analyze_radiology(self, image_path):
        if self.model is None:
            return {"error": "Model not loaded"}

        image_obj = self._get_image(image_path)
        image = self.preprocess(image_obj).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Original CLIP features
            clip_features = self.model.encode_image(image)
            
            # Adapted medical features
            adapted_features = self.adaptor(clip_features.float())
            
            # Normalize
            query_emb = adapted_features / adapted_features.norm(p=2, dim=-1, keepdim=True)
            q_vec = query_emb.cpu().numpy().astype('float32')

        # 1. FAISS Similarity Search
        top_matches = []
        if self.index:
            D, I = self.index.search(q_vec, 5)
            local_votes = {"positive": 0.0, "negative": 0.0}
            mv = {"chest": 0, "mri": 0, "ct": 0}

            for i, idx in enumerate(I[0]):
                lbl = self.metadata['labels'][idx].lower()
                sim = float(D[0][i])
                top_matches.append({
                    "path": self.metadata['paths'][idx].replace('\\', '/'),
                    "label": lbl,
                    "similarity": sim
                })
                
                # Modality and Diag votes
                if "chest" in lbl or "pneumonia" in lbl: mv["chest"] += 1
                elif "mri" in lbl: mv["mri"] += 1
                elif "ct" in lbl: mv["ct"] += 1
                
                vote_weight = 1.0 / (i + 1)
                if "pneumonia" in lbl or "infected" in lbl: local_votes["positive"] += vote_weight
                else: local_votes["negative"] += vote_weight
            
            modality = max(mv, key=mv.get)
        else:
            modality = "chest" # Default fallback
            local_votes = {"positive": 0.5, "negative": 0.5}

        # 2. Modality Validation
        valid, msg = self.validate_image(image_path, "radiology")
        if not valid:
            return {"error": msg}

        # 3. Hybrid Diagnosis Scoring
        def get_binary_sim(pos, neg):
            t = clip.tokenize([pos, neg]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_text(t)
                f = f / f.norm(p=2, dim=-1, keepdim=True)
            s = np.dot(q_vec[0], f.cpu().numpy().T)
            e_x = np.exp(s * 40.0)
            return e_x / e_x.sum()

        faiss_pos_ratio = local_votes["positive"] / (local_votes["positive"] + local_votes["negative"] + 1e-6)

        if modality == "chest":
            p = get_binary_sim("chest x-ray with pneumonia", "normal healthy chest x-ray")
            final_pos = (p[0] * 0.4) + (faiss_pos_ratio * 0.6)
            diag = "chest_pneumonia" if final_pos > 0.5 else "chest_normal"
            probs = {"pneumonia": float(final_pos), "normal": 1.0 - float(final_pos)}
        elif modality == "mri":
            p = get_binary_sim("abnormal pathology on brain MRI", "normal healthy brain MRI")
            final_pos = (p[0] * 0.4) + (faiss_pos_ratio * 0.6)
            diag = "mri_infected" if final_pos > 0.5 else "mri_normal"
            probs = {"abnormal": float(final_pos), "normal": 1.0 - float(final_pos)}
        else:
            p = get_binary_sim("infected abnormal abdominal CT", "normal healthy abdominal CT")
            final_pos = (p[0] * 0.4) + (faiss_pos_ratio * 0.6)
            diag = "ct_infected" if final_pos > 0.5 else "ct_normal"
            probs = {"infected": float(final_pos), "normal": 1.0 - float(final_pos)}

        # 4. Captions (Consistent with Modality & Diagnosis)
        cap_sims = np.dot(q_vec[0], self.caption_embeddings.T)
        
        # Apply logic-based weights to ensure consistency
        is_abnormal = "pneumonia" in diag or "infected" in diag or "abnormal" in diag
        
        for i, cap in enumerate(CANDIDATE_CAPTIONS):
            cap_lower = cap.lower()
            
            # Modality Mismatch Penalty (Broadened)
            if modality == "chest":
                if any(w in cap_lower for w in ["ct", "mri", "abdominal", "brain", "organs"]):
                    cap_sims[i] -= 2.0
            elif modality == "mri":
                if any(w in cap_lower for w in ["x-ray", "ct", "lungs", "pneumonia", "abdominal", "organs"]):
                    cap_sims[i] -= 2.0
            elif modality == "ct":
                if any(w in cap_lower for w in ["x-ray", "mri", "lungs", "pneumonia", "brain"]):
                    cap_sims[i] -= 2.0
                
            # Diagnosis Mismatch Penalty
            neg_words = ["pneumonia", "infection", "infected", "abnormal", "pathology"]
            pos_words = ["normal", "healthy", "clear"]
            
            if is_abnormal:
                if any(w in cap_lower for w in pos_words):
                    cap_sims[i] -= 1.5
            else:
                if any(w in cap_lower for w in neg_words):
                    cap_sims[i] -= 1.5

        top_cap = [CANDIDATE_CAPTIONS[i] for i in np.argsort(cap_sims)[::-1][:3]]

        return {
            "status": "success",
            "modality": modality.capitalize(),
            "diagnosis": diag.replace('_', ' ').capitalize(),
            "probabilities": probs,
            "topMatches": top_matches[:3],
            "predictedCaptions": top_cap,
            "groundTruth": DETAILED_TRUTH.get(diag),
            "confidenceScore": float(np.max(list(probs.values())))
        }

    def analyze_general(self, image_path):
        """Used for Pathology and Dynamics fallback"""
        if self.model is None: return torch.randn(1, 512)
        image_obj = self._get_image(image_path)
        image = self.preprocess(image_obj).unsqueeze(0).to(self.device)
        with torch.no_grad():
            clip_features = self.model.encode_image(image)
            adapted_features = self.adaptor(clip_features.float())
        return adapted_features

    def get_similarity(self, image_features, text_queries):
        if self.model is None: return {q: 0.5 for q in text_queries}
        text_tokens = clip.tokenize(text_queries).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return dict(zip(text_queries, similarity[0].tolist()))

    def validate_image(self, image_path, expected_modality):
        """Unified validation to prevent non-medical or incorrect uploads"""
        if self.model is None: return True, ""
        
        try:
            image_obj = self._get_image(image_path)
            image = self.preprocess(image_obj).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(image)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                f_vec = feat.cpu().numpy().astype('float32')

            v_targets = [
                "a medical radiology scan (x-ray, ct, or mri)",
                "a pathology slide or tissue microscopy image",
                "an ultrasound or endoscopy video frame",
                "a photo of an animal, pet, or wildlife",
                "a photo of a person, face, or human activity",
                "a photo of an everyday object, furniture, or tool",
                "a screenshot of text, website, or document",
                "a graphic, illustration, or digital artwork"
            ]
            v_tokens = clip.tokenize(v_targets).to(self.device)
            with torch.no_grad():
                vf = self.model.encode_text(v_tokens)
                vf = vf / vf.norm(p=2, dim=-1, keepdim=True)
            
            v_sims = np.dot(f_vec[0], vf.cpu().numpy().T)
            v_probs = np.exp(v_sims * 35.0) 
            v_probs /= v_probs.sum()
            
            # Diagnostic Logging
            print(f"\n--- Validation Check: {os.path.basename(image_path)} ---")
            print(f"Medical (Rad/Path/Dyn): {v_probs[0]:.4f} / {v_probs[1]:.4f} / {v_probs[2]:.4f}")
            print(f"Non-Medical (Animal/Person/Object/Text/Art): {v_probs[3]:.4f} / {v_probs[4]:.4f} / {v_probs[5]:.4f} / {v_probs[6]:.4f} / {v_probs[7]:.4f}")

            # 1. Primary Check: Is any non-medical category stronger than the medical ones?
            medical_score = np.sum(v_probs[:3])
            non_medical_score = np.sum(v_probs[3:])
            
            if non_medical_score > medical_score or np.max(v_probs[:3]) < 0.2:
                print(">>> REJECTED: Non-medical image detected.")
                return False, "Incorrect Format: The system detected a non-medical image. Please upload a valid clinical scan."

            # 2. Specific checks based on the active tab/endpoint
            if expected_modality == "radiology":
                if v_probs[0] < 0.35:
                    print(">>> REJECTED: Wrong medical modality (Radiology expected).")
                    return False, "Incorrect Modality: This appears to be a medical image, but not a Radiology scan (X-ray/CT/MRI)."
            elif expected_modality == "pathology":
                if v_probs[1] < 0.35:
                    print(">>> REJECTED: Wrong medical modality (Pathology expected).")
                    return False, "Incorrect Modality: This appears to be a medical image, but not a Pathology/WSI slide."
            elif expected_modality == "dynamics":
                if v_probs[2] < 0.35:
                    print(">>> REJECTED: Wrong medical modality (Ultrasound/Endoscopy expected).")
                    return False, "Incorrect Format: Please upload an Ultrasound or Endoscopy stream/image."

            print(">>> ACCEPTED: Valid medical image.")
            return True, ""
        except Exception as e:
            print(f"Validation Error: {e}")
            return False, f"Process Error: Could not read image file. {str(e)}"

    def _get_image(self, file_path):
        """Helper to load image or extract frame from video"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame)
                else:
                    raise ValueError("Could not read video stream.")
            except ImportError:
                # Fallback if cv2 failed to install properly
                raise ImportError("OpenCV is required for video analysis.")
        return Image.open(file_path)

_model_instance = None
def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = UniMedModel()
    return _model_instance
