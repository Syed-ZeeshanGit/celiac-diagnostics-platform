# 🔬 Clinical Image Diagnostics & Analytics Platform
A multi-disciplinary medical software ecosystem uniting **Vectorized Machine Learning Algorithms**, **Rigorous Data Hygiene Architectural Principles**, and **Clinical Decision Support Workflow Automation**.

### 🌐 Live Deployment
* **Interactive Production Dashboard:** [celiac-diagnostics-platform.streamlit.app](https://celiac-diagnostics-platform-jjnmmejmxsobz4khqcbgps.streamlit.app/)

---

### 🎯 Project Objective
The core objective of this project is to model and analyze the diagnostic criteria of Celiac Disease to build a reliable, non-invasive risk-stratification pipeline. Specifically, the system analyzes the predictive power of a patient's clinical phenotypic presentations (such as chronic diarrhoea, abdominal distress, idiopathic short stature, and unexplained weight loss) combined with quantitative serology blood biomarkers (IgA, IgG, and IgM antibodies). By mapping these pre-diagnostic features to historical patient outcomes, the platform isolates the exact baseline variables that indicate high risk, creating an automated tracking tool to determine whether an individual requires invasive confirmatory endoscopy and tissue biopsy.

### 🔄 System Architecture & Process
The platform isolates concerns cleanly across three distinct, curriculum-aligned academic pillars:

1. **(Data Hygiene & Anti-Leakage Pipeline):** Establishes a deterministic parsing pipeline that handles data cleaning, binary parameter mapping, and sparse one-hot matrix encoding. Crucially, the pipeline explicitly drops downstream diagnostic indicators (`Marsh` tissue grading score and `cd_type` classifications) from the training matrix. This avoids mathematical **Data Leakage**, ensuring the system generalizes safely to real-world clinical contexts where only pre-diagnostic phenotypic data is accessible.
2. **(Algorithmic Modeling & Image Processing):** * *Tabular Modeling:* Implements a pure, vectorized **Logistic Regression with Gradient Descent** from scratch. The model minimizes a Binary Cross-Entropy loss function iteratively using raw matrix dot products.
   * *Vision Modeling:* Implements a native **2D Max-Pooling Downsampling Engine** utilizing custom nested loops to compute regional structural feature map selections over grayscale biopsy image matrices ($O(N \times M)$ runtime complexity).
3. **(Triage Engineering & Auditing):** Connects the backend prediction vectors to an interactive patient intake portal. It evaluates classification stability metrics (Sensitivity, Specificity, and $F_1$-score) and writes transactional diagnostic evaluations to a secure, immutable operational tracking logger modeled after HIPAA/GDPR compliance standards.

---

### 📊 Project Outcomes & Validation
* **High-Accuracy Screening Performance:** Achieved an out-of-sample prediction performance of **$96.15\%$ Accuracy**, **$98.43\%$ Sensitivity (True Positive Rate)**, and an **$F_1$-score of $97.78\%$** using strictly non-invasive phenotypic features and serology antibodies.
* **Algorithmic Transparency:** Integrated a live mathematical verification section embedding a real-time visualization of the gradient descent convex cost function curve decaying across training iterations.
* **Full Stack Medical Utility:** Delivered an interactive clinical portal linking automated statistical triage assessments directly to image transformation modules and systemic auditing frameworks.
