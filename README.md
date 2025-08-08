 Logistic Regression - Breast Cancer Detection

This project uses logistic regression to predict if a tumor is cancerous (malignant) or not (benign) using the Breast Cancer dataset.



 Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

 Dataset

- File: `data.csv`
- Target column: `diagnosis`
  - M = Malignant (1)
  - B = Benign (0)

 Steps Done

1. Loaded and cleaned the dataset.
2. Converted target to numbers (M → 1, B → 0).
3. Split data into training and test sets.
4. Standardized features.
5. Trained a logistic regression model.
6. Evaluated using:
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC Curve
7. Saved the result images.

 Output Images

These are saved:
- `confusion_matrix.png`
- `roc_curve.png`


   ```bash
   pip install pandas scikit-learn matplotlib seaborn
