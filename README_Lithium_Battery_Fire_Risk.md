# 🔥 Deep Learning-Based Prediction of Lithium Battery Fire Risk

This project develops an **Artificial Neural Network (ANN)** to predict whether a fire incident involves **lithium-ion batteries**, using **real-world emergency incident data** from the **London Fire Brigade**. The goal is to build a proactive model to assist public safety teams in identifying high-risk scenarios early.

---

## 📌 Project Overview

> Fire risk associated with lithium-ion batteries in electric vehicles (EVs) is rising due to thermal runaway, overcharging, and poor design. This project builds a deep learning model to classify such incidents and assist emergency services.

### Key Objectives:
- Build an ANN classifier using structured emergency report data
- Predict if lithium-ion batteries are involved in an incident
- Provide balanced and generalizable predictions for risk detection

---

## 🛠️ Tools & Technologies

- **Language**: Python (TensorFlow, Scikit-learn, Pandas)
- **Model**: Feedforward Artificial Neural Network (ANN)
- **Methodology**: CRISP-DM
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: London Fire Brigade Incident Records

---

## 🗂️ Project Files

| File | Description |
|------|-------------|
| `DEEP LEARNING PROJECT.ipynb` | Jupyter notebook implementing data prep, model training, evaluation |
| `Deep learning Report.docx` | Full academic report (includes methodology, results, references) |
| `Lithium Battery Fire Risk Presentation.pptx` | Slide deck summarizing project |
| `Deep learning.pdf` | Final version of report in PDF format |

---

## 📊 Model Architecture

| Layer | Details |
|-------|---------|
| Input | Preprocessed features (Label Encoded + Scaled) |
| Dense 1 | 64 neurons, ReLU activation |
| Dropout | 30% |
| Dense 2 | 32 neurons, ReLU activation |
| Dropout | 20% |
| Output | 1 neuron, Sigmoid activation (Binary Classification) |

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 50 (Early Stopping applied)
- Validation Split: 16%

---

## 📈 Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 84.76% |
| Precision | 82% |
| Recall | 81% |
| F1-Score | 81% |

- **Balanced Performance**: Confusion matrix shows low false positives and false negatives (23 FP, 25 FN)
- **Training Accuracy**: ~87%
- **Validation Accuracy**: ~78–80%
- **Loss Curves**: Show consistent learning and no overfitting

---

## 📚 Dataset Details

- **Source**: [London Fire Brigade – London Data Store](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)
- **Features Used**: Ignition source, power type, vehicle manufacturer, etc.
- **Target Variable**: Whether lithium-ion batteries were involved in the fire

---

## 💡 Methodology: CRISP-DM

1. **Business Understanding**: Predict battery fire incidents for early response.
2. **Data Understanding**: Exploratory analysis on structured emergency reports.
3. **Data Preparation**:
   - Handle missing values
   - Label encoding of categorical fields
   - Feature scaling (StandardScaler)
4. **Modeling**: Feedforward ANN with dropout regularization
5. **Evaluation**: Accuracy, confusion matrix, precision-recall, F1-score
6. **Deployment Plan**: Lightweight model suitable for real-time classification systems

---

## 🧠 Conclusion & Contributions

- ✅ First ANN-based proactive model for lithium-ion fire incident classification
- 🔁 Balanced generalization across both classes (battery/non-battery)
- 🚀 Lightweight, retrainable, and deployable architecture

### Future Work:
- Integrate time-series sensor data from EVs
- Add model explainability (SHAP / LIME)
- Deploy via dashboard or emergency risk management system

---

## 👨‍💻 Contributors

- **Utkarsh Satpute** – Data Preparation, ANN Modeling, Report Writing
- **Tushar Gharpure** – Evaluation Metrics, Visualizations, Presentation
- **Pintoo Baghel** – Literature Review, Model Tuning, Future Work

---

## 📘 License

This project was submitted for academic evaluation under **Deep Learning and Generative AI** module at **National College of Ireland**. For educational use only.

---

## 🔖 Tags

`#LithiumBattery` `#FireRisk` `#DeepLearning` `#ANN` `#EVs` `#SafetyPrediction` `#CRISPDM` `#LondonFireBrigade` `#RealWorldData` `#TensorFlow`