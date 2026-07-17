# Building Machine Learning Pipeline on Startup Acquisition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [About the Project](#about-the-project)
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## About the Project

This project aims to **understand the financial conditions of company fundraising goals** by building a complete Machine Learning pipeline to predict the acquisition status of startups. Given historical data about startups — including their funding history, milestones, categories, and geographic information — the model predicts whether a startup is currently **Operating**, **IPO**, **Acquired**, or **Closed**.

## Objective

The objective is to predict the status of a startup using a **Supervised Machine Learning** approach. The model is trained on historical data of startups that were either acquired, closed, went through IPO, or are still operating. This provides actionable insights into the financial trajectory and viability of startups.

## Dataset

- **Source**: [Google Drive — Raw Data (JSON & Excel)](https://drive.google.com/file/d/1tWYkHYHm2HoiCajZ49Cs1K7sklWTdAbV/view)
- **Kaggle Reference**: `kaggle kernels output harsh9759/acquisition-status-self-scored-python-companies -p /path/to/dest`

### Summary

The dataset contains **industry trends, investment insights, and individual company information**. It includes 44 columns covering company metadata, funding details, milestones, and status labels. After preprocessing, a curated subset of features is used for model training.

### Key Features

| Feature | Description |
|---|---|
| `status` | Target variable — Operating, IPO, Acquired, or Closed |
| `funding_total_usd` | Total funding received in USD |
| `funding_rounds` | Number of funding rounds |
| `founded_at` | Date the company was founded |
| `category_code` | Industry category of the startup |
| `country_code` | Country where the startup is based |
| `milestones` | Number of milestones achieved |
| `relationships` | Number of relationships/connections |
| `first_funding_at` / `last_funding_at` | First and last funding dates |
| `first_milestone_at` / `last_milestone_at` | First and last milestone dates |

> **Note**: The raw dataset is large (JSON/Excel). It is **not** included in this repository. Download it from the Google Drive link above and place `companies.csv` in the project root before running the notebooks.

## Project Structure

```
Building-Machine-Learning-Pipeline-on-Startup-Acquisition/
│
├── README.md                                  # This file
├── PROJECT_DOCUMENTATION.md                   # Detailed technical documentation
├── requirements.txt
│
├── Startup_Steps_Data_Preprocessing.ipynb     # Step 1: cleaning & feature prep
├── ML_Pipeline_on_Startup_Acquisition.ipynb   # Step 2: EDA + model training/eval
│
├── src/
│   └── startup_pipeline.py                     # The sklearn training pipeline
├── tests/
│   └── test_pipeline.py                        # Offline unit tests (synthetic data)
│
└── companies.csv                              # Dataset (NOT tracked — download separately)
```

## Pipeline Overview

```
Raw Data (companies.csv)
        │
        ▼
┌─────────────────────────────┐
│  1. DATA PREPROCESSING      │  ← Startup_Steps_Data_Preprocessing.ipynb
│  • Remove irrelevant cols   │
│  • Handle missing values    │
│  • Remove duplicates        │
│  • Outlier detection (IQR)  │
│  • Date transformation      │
│  • Feature engineering      │
│  • One-hot encoding         │
│  • Save cleaned_companies   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. EDA + MODEL TRAINING    │  ← ML_Pipeline_on_Startup_Acquisition.ipynb
│  • Missing-data & corr maps │
│  • Outlier removal (numeric)│
│  • sklearn Pipeline:        │
│    impute → scale/one-hot   │
│    (fit on train only)      │
│  • LogReg + RandomForest    │
│    (balanced class weights) │
│  • accuracy / macro-F1 /    │
│    confusion matrix         │
└─────────────────────────────┘
        │
        ▼   src/startup_pipeline.py  (importable, unit-tested)
```

The target is the dataset's real `status` column (operating / acquired /
closed / ipo). Encoding, imputation, and scaling happen **inside** the sklearn
`Pipeline`, fit on the training split only, so there is no train/test leakage.

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Jupyter Notebook** or **VS Code** with Jupyter extension
- The following Python libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
category_encoders
scipy
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Building-Machine-Learning-Pipeline-on-Startup-Acquisition.git
   cd Building-Machine-Learning-Pipeline-on-Startup-Acquisition
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Download from [Google Drive](https://drive.google.com/file/d/1tWYkHYHm2HoiCajZ49Cs1K7sklWTdAbV/view)
   - Place `companies.csv` in the project root directory

## Usage

Run the notebooks in order:

```bash
# Step 1: cleaning & feature prep
jupyter notebook Startup_Steps_Data_Preprocessing.ipynb

# Step 2: EDA + model training and evaluation
jupyter notebook ML_Pipeline_on_Startup_Acquisition.ipynb
```

Or train straight from the command line (no notebook needed):

```bash
python -m src.startup_pipeline --data companies.csv --target status
```

Run the offline unit tests (synthetic data — no dataset required):

```bash
pytest -q
```

## Results

Preprocessing and EDA:
- Cleaned the 44-column Crunchbase export; engineered `age_in_days`, `founded_year`.
- Correlation analysis and (numeric-only) z-score outlier removal.

Modeling — predict `status` (operating / acquired / closed / ipo) with a
leak-free sklearn Pipeline, comparing two balanced classifiers:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression (baseline) | _fill from your run_ | _fill_ |
| Random Forest | _fill from your run_ | _fill_ |

> Run Step 2 (or `python -m src.startup_pipeline`) to generate these numbers,
> then paste them here. **Macro-F1 is the honest headline** — the `status`
> classes are imbalanced (mostly "operating"), so raw accuracy is inflated by
> the majority class.

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-Learn** | Machine learning (LabelEncoder, preprocessing) |
| **Category Encoders** | Advanced encoding (Binary, Target encoding) |
| **SciPy** | Statistical functions (Z-score outlier detection) |
| **Jupyter Notebook** | Interactive development environment |

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure your code follows the existing notebook structure and includes appropriate comments.

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

## Acknowledgments

- [CrunchBase](https://www.crunchbase.com/) — Original data source for startup information
- [Kaggle](https://www.kaggle.com/) — Community reference (harsh9759/acquisition-status)
- Scikit-Learn documentation for preprocessing utilities
- Seaborn and Matplotlib documentation for visualization techniques
