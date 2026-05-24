# src/clinical_flow.py
import numpy as np
from typing import Dict, Any, List

class ClinicalValidationEngine:
    """
    Evaluates classification reliability matrices to ensure diagnostic safety thresholds.
    """
    @staticmethod
    def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        return {
            "Sensitivity (TPR)": float(sensitivity),
            "Specificity (TNR)": float(specificity),
            "Precision": float(precision),
            "F1-Score Model Balance": float(f1)
        }


class DiagnosticAuditJournal:
    """
    Simulates secure institutional logging for HIPAA/GDPR clinical compliance tracking.
    """
    def __init__(self):
        self.journal_records: List[Dict[str, Any]] = []

    def sign_record(self, age: int, gender: str, probability: float, outcome: str) -> None:
        import time
        self.journal_records.append({
            "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "Patient Demographics": f"Age: {age} | Gender: {gender}",
            "Risk Probability Score": f"{probability * 100:.2f}%",
            "Triage Action Path": "Flagged For Biopsy" if outcome == "Positive" else "Discharged to Baseline Monitoring"
        })