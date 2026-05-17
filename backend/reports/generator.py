from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class MedicalReportGenerator:
    def __init__(self, model_name="microsoft/biogpt", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Medical LLM ({model_name}) on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)
        except Exception as e:
            print(f"Error loading LLM: {e}. Falling back to template-based reasoning.")
            self.generator = None

    def generate_findings(self, diagnosis, confidence, modality, similar_cases=None):
        """
        Generate dynamic medical findings based on classification and retrieval.
        """
        prompt = f"Context: Modality: {modality}, Diagnosis: {diagnosis}, Confidence: {confidence:.2f}%. "
        prompt += "Instruction: Write a clinical-style medical finding for this case. Focus on specific observations. "
        
        if self.generator:
            output = self.generator(prompt, max_length=150, num_return_sequences=1, do_sample=True)
            return output[0]['generated_text'].replace(prompt, "").strip()
        else:
            # Fallback reasoning
            return self._template_reasoning(diagnosis, modality)

    def _template_reasoning(self, diagnosis, modality):
        templates = {
            "Pneumonia": "Consolidation and focal opacities are observed in the lung fields, consistent with infectious processes. Clinical correlation is advised.",
            "Normal": "No significant acute abnormalities identified. The anatomical structures appear within normal limits for this modality.",
            "Brain Tumor": "A localized mass lesion with associated edema is identified. Signal characteristics suggest a neoplastic process.",
            "Infection": "Localized areas of increased density and inflammatory markers are present, suggesting an active infection site.",
            "Cancerous tissue": "Malignant cell morphology and architectural distortion are observed in the tissue section.",
            "Normal tissue": "Tissue architecture is preserved with no evidence of malignancy or significant atypia."
        }
        return templates.get(diagnosis, f"Clinical observations for {diagnosis} are noted with high confidence in the {modality} study.")

    def final_structured_report(self, data):
        report = f"""
================================================
MEDICAL AI ANALYSIS REPORT
================================================

Modality:
{data['modality']}

Prediction:
{data['prediction']}

Confidence:
{data['confidence']}%

Findings:
{data['findings']}

AI Explanation:
{data['explanation']}

Recommendation:
Clinical correlation and physician review advised.
================================================
"""
        return report
