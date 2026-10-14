# Task 3 — Real-World Dataset Analysis Project

## Dataset
This project uses the diabetes regression dataset distributed with scikit-learn (`sklearn.datasets.load_diabetes`). The target is a quantitative measure of disease progression one year after baseline, **not** a diagnosis of diabetes. The distributed feature values are standardized.

Source documentation: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

## Important data note
`diabetes_raw.csv` is a working copy of the real dataset with a small number of missing feature values and three duplicate observations intentionally introduced to demonstrate cleaning. These quality issues were added for this educational project and are not claims about the untouched source dataset.

## Files
- `Task3_Real_World_Dataset_Analysis_Report.docx` — detailed report with methods, tables, plots, findings, and limitations.
- `diabetes_raw.csv` — raw working copy.
- `diabetes_cleaned.csv` — cleaned data used in the analysis.
- `task3_analysis.py` — reproducible cleaning and analysis script.
- `outputs/` — data quality summary, descriptive statistics, correlation matrix, target correlations, and quartile summary.
- `plots/` — visualizations included in the report.
- `task3_submission_description.txt` — description to paste into the internship portal.

## Run
Install Python packages if needed:
```bash
pip install pandas numpy matplotlib scikit-learn
```
Run:
```bash
python task3_analysis.py
```
The script uses the CSV included in this folder and does not require downloading the dataset.

## Main steps
1. Inspect shape, columns, missing values, and duplicates.
2. Remove duplicate observations while excluding the unique record ID.
3. Convert analysis columns to numeric and median-impute missing numeric values.
4. Export cleaned data, descriptive statistics, and correlations.
5. Create a target distribution plot.

## Limitations
The dataset is small, feature values are standardized, and correlation does not imply causation. The report is an educational exploratory analysis and is not a clinical tool.
