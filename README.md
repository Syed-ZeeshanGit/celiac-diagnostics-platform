# 🔬 Clinical Image Diagnostics & Analytics Platform
A multi-disciplinary medical software ecosystem uniting **Vectorized Machine Learning Algorithms**, **Rigorous Data Hygiene Architectural Principles**, and **Clinical Decision Support Workflow Automation**.

### 🌐 Live Deployment
* **Interactive Production Dashboard:** [celiac-diagnostics-platform.streamlit.app](https://celiac-diagnostics-platform-jjnmmejmxsobz4khqcbgps.streamlit.app/)

---

### 🎯 Project Objective
The primary objective of this project is to build an end-to-end clinical triage and diagnostics processing pipeline for Celiac Disease. By implementing the mathematical modeling and matrix transformations completely from scratch using raw `NumPy` and `Pandas` (completely avoiding high-level frameworks like `scikit-learn` or `TensorFlow`), the platform serves as an explicit demonstration of foundational core computer science, software engineering hygiene, and healthcare information systems integration.

### 🔄 System Architecture & Process
The platform isolates concerns cleanly across three distinct, curriculum-aligned academic pillars:

1. **Software Engineering (Data Hygiene & Anti-Leakage Pipeline):** Establishes a deterministic parsing pipeline that handles data cleaning, binary parameter mapping, and sparse one-hot matrix encoding. Crucially, the pipeline explicitly drops downstream diagnostic indicators (`Marsh` tissue grading score and `cd_type` classifications) from the training matrix. This avoids mathematical **Data Leakage**, ensuring the system generalizes safely to real-world clinical contexts where only pre-diagnostic phenotypic data is accessible.
2. **Computer Science (Algorithmic Modeling & Image Processing):** * *Tabular Modeling:* Implements a pure, vectorized **Logistic Regression with Gradient Descent** from scratch. The model minimizes a Binary Cross-Entropy loss function iteratively using raw matrix dot products.
   * *Vision Modeling:* Implements a native **2D Max-Pooling Downsampling Engine** utilizing custom nested loops to compute regional structural feature map selections over grayscale biopsy image matrices ($O(N \times M)$ runtime complexity).
3. **Information Systems (Triage Engineering & Auditing):** Connects the backend prediction vectors to an interactive patient intake portal. It evaluates classification stability metrics (Sensitivity, Specificity, and $F_1$-score) and writes transactional diagnostic evaluations to a secure, immutable operational tracking logger modeled after HIPAA/GDPR compliance standards.

---

### 📊 Project Outcomes & Validation
* **High-Accuracy Screening Performance:** Achieved an out-of-sample prediction performance of **$96.15\%$ Accuracy**, **$98.43\%$ Sensitivity (True Positive Rate)**, and an **$F_1$-score of $97.78\%$** using strictly non-invasive phenotypic features and serology antibodies.
* **Algorithmic Transparency:** Integrated a live mathematical verification section embedding a real-time visualization of the gradient descent convex cost function curve decaying across training iterations.
* **Full Stack Medical Utility:** Delivered an interactive clinical portal linking automated statistical triage assessments directly to image transformation modules and systemic auditing frameworks.
