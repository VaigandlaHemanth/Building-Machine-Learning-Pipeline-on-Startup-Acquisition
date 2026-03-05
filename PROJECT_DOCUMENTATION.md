# Building Machine Learning Pipeline on Startup Acquisition — Detailed Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Flow — High-Level Architecture](#2-project-flow--high-level-architecture)
3. [Workflow Description — Step-by-Step](#3-workflow-description--step-by-step)
4. [Detailed Component Breakdown](#4-detailed-component-breakdown)
   - [4.1 Data Loading & Initial Exploration](#41-data-loading--initial-exploration)
   - [4.2 Data Cleaning — Irrelevant & Redundant Column Removal](#42-data-cleaning--irrelevant--redundant-column-removal)
   - [4.3 Duplicate Removal](#43-duplicate-removal)
   - [4.4 Null Value Handling](#44-null-value-handling)
   - [4.5 Outlier Detection & Removal](#45-outlier-detection--removal)
   - [4.6 Date Transformation](#46-date-transformation)
   - [4.7 Categorical Variable Generalization](#47-categorical-variable-generalization)
   - [4.8 Feature Engineering](#48-feature-engineering)
   - [4.9 Data Encoding](#49-data-encoding)
   - [4.10 Correlation Analysis & Feature Selection](#410-correlation-analysis--feature-selection)
   - [4.11 Visualization & EDA](#411-visualization--eda)
   - [4.12 Data Labeling](#412-data-labeling)
   - [4.13 Final Cleaned Dataset Export](#413-final-cleaned-dataset-export)
5. [Optimizers & Algorithms — Detailed Explanation](#5-optimizers--algorithms--detailed-explanation)
   - [5.1 IQR-Based Outlier Detection Algorithm](#51-iqr-based-outlier-detection-algorithm)
   - [5.2 Z-Score Outlier Detection Algorithm](#52-z-score-outlier-detection-algorithm)
   - [5.3 Correlation-Based Feature Selection Algorithm](#53-correlation-based-feature-selection-algorithm)
   - [5.4 Encoding Algorithms](#54-encoding-algorithms)
   - [5.5 Chunking / Generalization Algorithm for High-Cardinality Categoricals](#55-chunking--generalization-algorithm-for-high-cardinality-categoricals)
   - [5.6 Null Value Threshold Filtering Algorithm](#56-null-value-threshold-filtering-algorithm)
6. [Data Types — Full Column Reference](#6-data-types--full-column-reference)
7. [Libraries & Dependencies](#7-libraries--dependencies)
8. [Notebook-by-Notebook Detailed Walkthrough](#8-notebook-by-notebook-detailed-walkthrough)
   - [8.1 Startup_Steps_Data_Preprocessing.ipynb](#81-startup_steps_data_preprocessingipynb)
   - [8.2 doing.ipynb](#82-doingipynb)
   - [8.3 ML_Pipeline_on_Startup_Acquisition.ipynb](#83-ml_pipeline_on_startup_acquisitionipynb)
9. [Why Each Decision Was Made](#9-why-each-decision-was-made)
10. [Summary](#10-summary)

---

## 1. Project Overview

**Title**: Building Machine Learning Pipeline on Startup Acquisition

**Purpose**: To understand the financial conditions of company fundraising goals and predict the acquisition status of startups.

**Target Audience**: Data scientists, ML engineers, business analysts, and investors seeking to understand and predict startup outcomes.

**Core Problem**: Given a startup's historical data (funding, milestones, category, geography, relationships), predict whether it is currently **Operating**, went through **IPO**, was **Acquired**, or is **Closed**.

**Approach**: Supervised Machine Learning — classification task trained on historical startup data.

**Key Features of the Project**:
- Complete end-to-end ML data pipeline (ingestion → cleaning → transformation → encoding → feature selection)
- Multiple encoding strategies demonstrated (Label, One-Hot, Binary, Frequency, Target)
- Robust outlier detection using both IQR and Z-score methods
- Correlation-based feature selection with configurable thresholds
- Comprehensive EDA with heatmaps, histograms, boxplots, pairplots, and scatterplots

**Technologies Used**:
- Python 3.8+
- Pandas (data manipulation)
- NumPy (numerical computing)
- Matplotlib (plotting)
- Seaborn (statistical visualization)
- Scikit-Learn (LabelEncoder, preprocessing)
- Category Encoders (Binary, Target encoding)
- SciPy (Z-score statistics)
- Jupyter Notebook (interactive environment)

---

## 2. Project Flow — High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PROJECT ARCHITECTURE                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   ┌──────────────────┐                                                     ║
║   │   RAW DATA       │   companies.csv (44 columns, ~196,000+ rows)       ║
║   │   (Google Drive)  │   Industry trends, investments, company info       ║
║   └────────┬─────────┘                                                     ║
║            │                                                               ║
║            ▼                                                               ║
║   ┌──────────────────────────────────────────────────────────────┐         ║
║   │     PHASE 1: DATA PREPROCESSING                              │         ║
║   │     (Startup_Steps_Data_Preprocessing.ipynb)                 │         ║
║   │                                                              │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │         ║
║   │  │ Col Removal │→│  Dedup      │→│ Null Handling     │    │         ║
║   │  │ (Irrelevant │  │ (Exact     │  │ (96% threshold   │    │         ║
║   │  │  Redundant  │  │  match)    │  │  + row dropping) │    │         ║
║   │  │  Granular)  │  │            │  │                  │    │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘    │         ║
║   │                                               │             │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌─────────▼────────┐   │         ║
║   │  │ Outlier     │→│ Date        │→│ Category         │   │         ║
║   │  │ Removal     │  │ Transform  │  │ Generalization   │   │         ║
║   │  │ (IQR)       │  │ (→ Years)  │  │ (Top 10 + other) │   │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘   │         ║
║   │                                               │             │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌─────────▼────────┐   │         ║
║   │  │ Feature     │→│ One-Hot     │→│ Mean Imputation  │   │         ║
║   │  │ Engineering │  │ Encoding   │  │ (remaining NaN)  │   │         ║
║   │  │ (isClosed,  │  │ (category  │  │                  │   │         ║
║   │  │  age_days)  │  │  +country) │  │                  │   │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘   │         ║
║   │                                               │             │         ║
║   │                                    ┌──────────▼───────┐    │         ║
║   │                                    │ cleaned_companies │    │         ║
║   │                                    │     .csv          │    │         ║
║   │                                    └──────────┬───────┘    │         ║
║   └────────────────────────────────────────────────┼────────────┘         ║
║                                                    │                      ║
║            ▼                                       │                      ║
║   ┌──────────────────────────────────────────────────────────────┐         ║
║   │     PHASE 2: FEATURE ENGINEERING & EDA                       │         ║
║   │     (doing.ipynb)                                            │         ║
║   │                                                              │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │         ║
║   │  │ NaN %       │→│ 45% thresh  │→│ Duplicate check  │    │         ║
║   │  │ Analysis    │  │ col drop    │  │                  │    │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘    │         ║
║   │                                               │             │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌─────────▼────────┐   │         ║
║   │  │ Date        │→│ Col Drop    │→│ NaN Fill         │   │         ║
║   │  │ Parsing     │  │ (id,logo,  │  │ (cat→Unknown     │   │         ║
║   │  │             │  │  name etc) │  │  num→0)          │   │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘   │         ║
║   │                                               │             │         ║
║   │  ┌──────────────────┐  ┌──────────────────────▼──────┐     │         ║
║   │  │ Label Encoding    │→│ Status Encoding             │     │         ║
║   │  │ (high cardinality │  │ (acquired=1,operating=0    │     │         ║
║   │  │  >100 unique)     │  │  closed=2, ipo=3)          │     │         ║
║   │  └──────────────────┘  └──────────────────────┬──────┘     │         ║
║   │                                               │             │         ║
║   │  ┌──────────────────┐  ┌──────────────────────▼──────┐     │         ║
║   │  │ Correlation      │→│ Feature Selection            │     │         ║
║   │  │ Matrix vs target │  │ (drop if |corr| < 0.009)   │     │         ║
║   │  └──────────────────┘  └──────────────────────┬──────┘     │         ║
║   │                                               │             │         ║
║   │  ┌──────────────────┐  ┌──────────────────────▼──────┐     │         ║
║   │  │ Histograms       │  │ Box Plots                   │     │         ║
║   │  │ (all features)   │  │ (outlier visual)            │     │         ║
║   │  └──────────────────┘  └─────────────────────────────┘     │         ║
║   └──────────────────────────────────────────────────────────────┘         ║
║                                                                            ║
║            ▼                                                               ║
║   ┌──────────────────────────────────────────────────────────────┐         ║
║   │     PHASE 3: ML PIPELINE — EDA + ENCODING + LABELING        │         ║
║   │     (ML_Pipeline_on_Startup_Acquisition.ipynb)               │         ║
║   │                                                              │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │         ║
║   │  │ Full EDA     │→│ Missing     │→│ Distribution     │    │         ║
║   │  │ (info,head,  │  │ Data        │  │ Histograms       │    │         ║
║   │  │  describe)   │  │ Heatmap     │  │                  │    │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘    │         ║
║   │                                               │             │         ║
║   │  ┌─────────────┐  ┌─────────────┐  ┌─────────▼────────┐   │         ║
║   │  │ Correlation  │→│ Pairplot    │→│ Outlier          │   │         ║
║   │  │ Matrix +     │  │ Scatterplot │  │ Detection        │   │         ║
║   │  │ Heatmap      │  │             │  │ (Box,Z,IQR)      │   │         ║
║   │  └─────────────┘  └─────────────┘  └──────────┬───────┘   │         ║
║   │                                               │             │         ║
║   │  ┌──────────────────┐  ┌──────────────────────▼──────┐     │         ║
║   │  │ Encoding Demo     │→│ Data Labeling               │     │         ║
║   │  │ (Label, One-Hot,  │  │ (category→type mapping     │     │         ║
║   │  │  Binary, Freq,    │  │  status label functions)   │     │         ║
║   │  │  Target)          │  │                            │     │         ║
║   │  └──────────────────┘  └─────────────────────────────┘     │         ║
║   └──────────────────────────────────────────────────────────────┘         ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Workflow Description — Step-by-Step

### Step 1: Data Ingestion
- The raw dataset `companies.csv` is loaded using `pd.read_csv()`
- The data contains **44 columns** with information about startups: names, URLs, funding details, milestones, categories, countries, dates, and status labels
- Initial inspection: `.shape`, `.info()`, `.head()`, `.describe()`, `.columns`

### Step 2: Data Cleaning (Irrelevant & Redundant Removal)
- **Granular columns** removed: `region`, `city` (too fine-grained for meaningful pattern extraction)
- **Redundant columns** removed: `id`, `Unnamed: 0.1`, `entity_type`, `entity_id`, `parent_id`, `created_by`, `created_at`, `updated_at`
- **Irrelevant columns** removed: `domain`, `homepage_url`, `twitter_username`, `logo_url`, `logo_width`, `logo_height`, `short_description`, `description`, `overview`, `tag_list`, `name`, `normalized_name`, `permalink`, `invested_companies`

### Step 3: Duplicate Removal
- Check for duplicates using `.duplicated().sum()`
- Remove all exact duplicate rows with `.drop_duplicates(inplace=True)`
- Verify removal

### Step 4: Null Value Handling
- Calculate null percentages across all columns
- **Drop columns** with >96% null values (keeping `closed_at` for feature engineering)
- **Drop rows** with missing values in critical columns: `status`, `country_code`, `category_code`, `founded_at`
- Alternative approach (doing.ipynb): Drop columns with >45% null values as a threshold

### Step 5: Outlier Detection & Removal
- **IQR Method** applied to `funding_total_usd` and `funding_rounds`
- Calculate Q1, Q3, IQR; define bounds as Q1 - 1.5×IQR and Q3 + 1.5×IQR
- Remove rows outside bounds
- **Z-Score Method** (ML_Pipeline notebook): Filter rows where |Z| > 3

### Step 6: Date Transformation
- Convert all date columns (`founded_at`, `closed_at`, `first_funding_at`, `last_funding_at`, `first_milestone_at`, `last_milestone_at`) from string to datetime, then extract year
- This reduces date granularity to year, which is more useful as a numeric feature

### Step 7: Categorical Variable Generalization (Chunking)
- For `category_code` (42 unique values): Keep top 10 most frequent categories, replace all others with `'other'`
- For `country_code` (161 unique values): Keep top 10 most frequent countries, replace all others with `'other'`
- This prevents one-hot encoding from creating hundreds of sparse columns

### Step 8: Feature Engineering
- **`isClosed`**: Binary feature — 1 if status is 'operating' or 'ipo', 0 if 'acquired' or 'closed'
- **`age_in_days`**: `closed_at - founded_at` (in years after date transformation)
- **`founded_year`**: Extracted from `founded_at` datetime
- Status binary encoding: operating/ipo → 1, acquired/closed → 0

### Step 9: Data Encoding
- **One-Hot Encoding**: Applied to `category_code` and `country_code` (after generalization) with `drop_first=True` to avoid multicollinearity
- **Label Encoding**: Applied to high-cardinality object columns (>100 unique values)
- **One-Hot Encoding**: Applied to remaining low-cardinality object columns
- **Status Target Encoding**: acquired=1, operating=0, closed=2, ipo=3 (for multiclass)
- Additional encoding demos: Binary Encoding, Frequency Encoding, Target Encoding

### Step 10: Correlation Analysis & Feature Selection
- Compute full correlation matrix on encoded DataFrame
- Calculate correlation of each feature against `status_label`
- **Drop features** where absolute correlation with target < 0.009
- Preserve essential columns: `status_label`, `status_closed`, `status_ipo`, `status_operating`

### Step 11: Fill Remaining NaN Values
- Categorical columns → fill with `'Unknown'`
- Numerical columns → fill with `0` (doing.ipynb) or `mean` (preprocessing notebook)

### Step 12: Visualization
- Histograms with KDE for all features
- Box plots for outlier visualization
- Missing data heatmap (viridis colormap)
- Correlation matrix heatmap (coolwarm colormap, annotated)
- Pairplot for multi-variable relationships
- Scatterplots for bivariate analysis

### Step 13: Export
- Save final cleaned dataset to `cleaned_companies.csv`

---

## 4. Detailed Component Breakdown

### 4.1 Data Loading & Initial Exploration

**Component Name**: Data Ingestion & EDA

**Functionality**: Loads the raw CSV file and performs initial exploratory data analysis to understand the shape, types, distributions, and quality of the data.

**Input**: `companies.csv` — raw dataset with 44 columns

**Output**: Printed summaries: shape tuple, dtypes, statistical summary, column list, first 5 rows

**Dependencies**: `pandas` library

**Code** (from `doing.ipynb`):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('companies.csv')
print(df.shape)
df.info()
df.head()
df.describe()
```

**Code** (from `Startup_Steps_Data_Preprocessing.ipynb`):
```python
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
warnings.filterwarnings('ignore')

company = pd.read_csv("companies.csv")
company.head()
company.describe()
company.columns
```

**Code** (from `ML_Pipeline_on_Startup_Acquisition.ipynb`):
```python
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('/content/companies.csv')
data.info()
data.head()
data.describe()
```

**Why**: Initial exploration is essential to understand data quality, identify which columns are useful, spot missing values early, and understand distributions before applying any transformations.

---

### 4.2 Data Cleaning — Irrelevant & Redundant Column Removal

**Component Name**: Column Pruning

**Functionality**: Systematically removes columns that do not contribute to the prediction task. Columns are categorized into three removal rationales:

**Category A — Granular columns** (too location-specific):
```python
company = company.drop(['region', 'city'], axis=1)
```
**Why removed**: Region and city create extremely high cardinality with no generalizable patterns. A model would overfit to specific city names rather than learning meaningful geographic signals. `state_code` is later also removed.

**Category B — Redundant metadata columns**:
```python
company = company.drop(['id', 'Unnamed: 0.1', 'entity_type', 'entity_id', 
                         'parent_id', 'created_by', 'created_at', 'updated_at'], axis=1)
```
**Why removed**:
- `id`, `entity_id`, `Unnamed: 0.1` — Index columns with no predictive value
- `entity_type` — Always "Company" (constant column = zero information)
- `parent_id` — Mostly null, indicating parent-child relationship rarely present
- `created_by`, `created_at`, `updated_at` — Database metadata, not business features

**Category C — Irrelevant text/URL columns**:
```python
company = company.drop(['domain', 'homepage_url', 'twitter_username', 'logo_url', 
                         'logo_width', 'logo_height', 'short_description', 'description', 
                         'overview', 'tag_list', 'name', 'normalized_name', 'permalink', 
                         'invested_companies'], axis=1)
```
**Why removed**: 
- URLs, logos, descriptions are unstructured text/media — not useful for tabular ML without NLP
- `name` and `normalized_name` are identifiers, not features
- `tag_list` is free-text — would require NLP processing
- `invested_companies` is a nested field

**In `doing.ipynb`** (alternative approach):
```python
df.drop(columns=['id', 'Unnamed: 0.1', 'name', 'permalink', 'homepage_url', 
                  'logo_url', 'normalized_name'], inplace=True)
```

**Input**: Full DataFrame with 44 columns  
**Output**: Reduced DataFrame with ~14-18 meaningful columns  
**Dependencies**: None (pure pandas operations)

---

### 4.3 Duplicate Removal

**Component Name**: Deduplication

**Functionality**: Identifies and removes exact duplicate rows from the dataset.

**Code**:
```python
company.duplicated().sum()       # Count duplicates
company.drop_duplicates(inplace=True)  # Remove them
company.duplicated().sum()       # Verify: should be 0
```

**Input**: DataFrame (post column-pruning)  
**Output**: Deduplicated DataFrame  
**Dependencies**: None

**Why**: Duplicate rows inflate sample size artificially, bias model training (some patterns get more weight), and distort statistical measures. Even a small number of duplicates can bias a classifier if they cluster in one class.

---

### 4.4 Null Value Handling

**Component Name**: Missing Value Strategy

**Functionality**: Multi-strategy approach to handle null values depending on the column type and percentage of nulls.

#### Strategy 1: Column-Level Threshold Dropping (Preprocessing Notebook)

```python
# Drop columns with >96% null values (except closed_at)
threshold = len(company) * 0.96
columns_to_drop = company.columns[company.isnull().sum() > threshold]
columns_to_drop = columns_to_drop.drop('closed_at', errors='ignore')
company = company.drop(columns=columns_to_drop)
```

**Why 96% threshold**: Columns with >96% missing data have so little actual data that imputation would essentially be fabricating values. The imputed values would dominate and create noise rather than signal. The `closed_at` column is preserved because it's needed for feature engineering (`age_in_days` calculation).

#### Strategy 2: Column-Level Threshold Dropping (doing.ipynb)

```python
threshold = 0.45
null_threshold = len(df) * threshold
df = df.dropna(thresh=null_threshold, axis=1)
```

**Why 45% threshold**: This is more aggressive — any column where fewer than 45% of rows have data is dropped. This ensures remaining columns have sufficient data density for meaningful analysis.

#### Strategy 3: Row-Level Dropping for Critical Columns

```python
company.dropna(subset=['status', 'country_code', 'category_code', 'founded_at'], inplace=True)
```

**Why**: These four columns are fundamental to the prediction task:
- `status` IS the target variable — cannot predict without it
- `country_code` and `category_code` are core features
- `founded_at` is needed for temporal calculations

Imputing these would introduce incorrect patterns into the model.

#### Strategy 4: Categorical Null Fill

```python
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].fillna('Unknown', inplace=True)
```

**Why 'Unknown'**: For categorical features, filling with a specific category 'Unknown' preserves the information that data was missing (which itself may be a signal) without introducing a fake category value.

#### Strategy 5: Numerical Null Fill

```python
# Approach A (doing.ipynb): Fill with 0
numerical_columns = df.select_dtypes(include=['number']).columns
for col in numerical_columns:
    df[col].fillna(0, inplace=True)

# Approach B (Preprocessing notebook): Fill with mean
company.fillna(company.mean(), inplace=True)
```

**Why 0 vs Mean**:
- **0**: Used when 0 is a semantically meaningful default (e.g., 0 funding rounds = never funded)
- **Mean**: Used when the assumption is that missing values are similar to the average (less biased for numerical distributions)

**Input**: DataFrame with null values  
**Output**: DataFrame with no null values  
**Dependencies**: `pandas`

---

### 4.5 Outlier Detection & Removal

**Component Name**: Outlier Handling

**Functionality**: Identifies and removes statistical outliers using two methods.

#### Method 1: IQR (Interquartile Range) Method

```python
# For funding_total_usd
Q1 = company['funding_total_usd'].quantile(0.25)
Q3 = company['funding_total_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers_total_usd = company[(company['funding_total_usd'] < lower_limit) | 
                              (company['funding_total_usd'] > upper_limit)]

# For funding_rounds
Q1 = company['funding_rounds'].quantile(0.25)
Q3 = company['funding_rounds'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers_rounds = company[(company['funding_rounds'] < lower_limit) | 
                           (company['funding_rounds'] > upper_limit)]

# Drop outliers
company = company[~company.index.isin(outliers_rounds.index)]
company = company[~company.index.isin(outliers_total_usd.index)]
```

**Why IQR was chosen**:
- IQR is **non-parametric** — it doesn't assume the data follows a normal distribution (funding data is typically right-skewed)
- The 1.5×IQR rule is a widely accepted statistical convention (Tukey's fences)
- It is robust to extreme outliers because Q1 and Q3 themselves are not affected by extreme values
- Applied specifically to `funding_total_usd` and `funding_rounds` because these financial metrics are most prone to extreme values (a few mega-funded startups can skew the entire dataset)

**The 1.5 multiplier**: John Tukey's original box-and-whisker paper established 1.5×IQR as the boundary for "mild outliers" and 3×IQR for "extreme outliers". 1.5 captures approximately 99.3% of data in a normal distribution.

#### Method 2: Z-Score Method (ML Pipeline Notebook)

```python
from scipy import stats
data = data[(np.abs(stats.zscore(data['twitter_username'])) < 3)]
```

**Why Z-Score**: 
- Z-Score measures how many standard deviations a value is from the mean
- A threshold of 3 means values >3σ from the mean are outliers
- This assumes approximately normal distribution
- Best for features where normal distribution holds

#### Method 3: IQR Capping (Alternative in ML Pipeline Notebook)

```python
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR
data['col'] = np.where(data['col'] > upper_limit, upper_limit, data['col'])
data['col'] = np.where(data['col'] < lower_limit, lower_limit, data['col'])
```

**Why Capping vs Dropping**: Instead of removing outlier rows (which loses other column data), capping replaces extreme values with the boundary values. This preserves sample size while reducing outlier impact.

**Input**: DataFrame with outlier values  
**Output**: DataFrame with outliers removed or capped  
**Dependencies**: `numpy`, `scipy.stats`, `pandas`

---

### 4.6 Date Transformation

**Component Name**: Temporal Feature Extraction

**Functionality**: Converts date strings into year integers for use as numerical features.

```python
# Convert all dates to year
company['founded_at'] = pd.to_datetime(company['founded_at']).dt.year
company['closed_at'] = pd.to_datetime(company['closed_at']).dt.year
company['first_funding_at'] = pd.to_datetime(company['first_funding_at']).dt.year
company['last_funding_at'] = pd.to_datetime(company['last_funding_at']).dt.year
company['first_milestone_at'] = pd.to_datetime(company['first_milestone_at']).dt.year
company['last_milestone_at'] = pd.to_datetime(company['last_milestone_at']).dt.year
```

**Alternative approach in `doing.ipynb`**:
```python
date_columns = ['last_milestone_at', 'first_milestone_at', 'founded_at']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

df['founded_year'] = df['founded_at'].dt.year
```

**Why extract year only**:
- Full datetime objects cannot be used directly in ML models (non-numeric)
- Month/day granularity adds noise — startup status rarely depends on the exact month of founding
- Year captures the most meaningful temporal component (market conditions, tech era, funding climate)
- Keeps the feature space manageable

**Input**: Columns with date strings (e.g., "2005-01-15")  
**Output**: Columns with integer years (e.g., 2005)  
**Dependencies**: `pandas.to_datetime`

---

### 4.7 Categorical Variable Generalization

**Component Name**: High-Cardinality Categorical Chunking

**Functionality**: Reduces the number of unique categories by keeping only the most frequent values and grouping the rest as 'other'.

#### For `category_code` (42 unique categories → 11):
```python
other_categories = company['category_code'].value_counts()[10:].index
company['category_code'] = company['category_code'].replace(to_replace=other_categories, value='other')
```

#### For `country_code` (161 unique countries → 11):
```python
other_countries = company['country_code'].value_counts()[10:].index
company['country_code'] = company['country_code'].replace(to_replace=other_countries, value='other')
```

**Why Top-10 Chunking**:
- One-hot encoding 42 categories creates 42 new columns; 161 countries creates 161 columns
- Most values are concentrated in the top categories (Pareto principle)
- Rare categories have too few samples to learn meaningful patterns
- Grouping rare categories as 'other' preserves the signal that a company belongs to a less common category
- Top-10 strikes a balance: captures the majority of data distribution while keeping dimensionality manageable

**Algorithm Details**:
1. Count value frequencies with `.value_counts()`
2. Sort descending (default)
3. Take index from position 10 onward (the less frequent ones)
4. Replace all those values with 'other'
5. Apply one-hot encoding after generalization

This is effectively a **frequency-based binning/chunking algorithm** — a form of dimensionality reduction for categorical variables.

**Input**: Categorical columns with high cardinality  
**Output**: Same columns with reduced cardinality (≤11 unique values)  
**Dependencies**: `pandas`

---

### 4.8 Feature Engineering

**Component Name**: New Feature Creation

**Functionality**: Creates derived features that capture business-relevant information not directly present in individual columns.

#### Feature: `isClosed`
```python
company['status'] = company['status'].replace('operating', 1)
company['status'] = company['status'].replace('ipo', 1)
company['status'] = company['status'].replace('closed', 0)
company['status'] = company['status'].replace('acquired', 0)

company['isClosed'] = company.apply(lambda row: 1 if row['status'] == 'closed' else 0, axis=1)
```

**Purpose**: Binary indicator of whether a company is currently closed. Simplifies the target into a binary signal for certain analyses.

#### Feature: `age_in_days` (actually in years after transformation)
```python
# Fill NaN in closed_at: operating/ipo companies → 2021 (data collection year)
for i in company['status']:
    if (i == 'operating' or 'ipo'):
        company['closed_at'].fillna(2021, inplace=True)
    elif (i == 'acquired' or 'closed'):
        company['closed_at'].fillna(0, inplace=True)

# Calculate age
company['age_in_days'] = (company['closed_at'] - company['founded_at'])
company.drop(['closed_at'], axis=1, inplace=True)
```

**Purpose**: Captures how long a company has been active. Longer active periods may correlate with success (operating/IPO). The `closed_at` column is then dropped since its information is captured in `age_in_days`.

#### Feature: `founded_year`
```python
df['founded_year'] = df['founded_at'].dt.year
```

**Purpose**: Explicit year feature for temporal analysis. Companies founded in different eras faced different market conditions.

**Input**: Existing columns (status, closed_at, founded_at)  
**Output**: New columns (isClosed, age_in_days, founded_year)  
**Dependencies**: `pandas`

---

### 4.9 Data Encoding

**Component Name**: Categorical to Numerical Conversion

**Functionality**: Transforms categorical/textual features into numerical representations for ML model consumption.

#### 4.9.1 One-Hot Encoding (Primary Method Used)

```python
# After generalization (Preprocessing notebook)
company = pd.get_dummies(company, columns=['category_code'], drop_first=True)
company = pd.get_dummies(company, columns=['country_code'], drop_first=True)
```

```python
# For low cardinality columns (doing.ipynb)
low_cardinality_columns = [col for col in df.select_dtypes(include=['object']).columns 
                            if col not in high_cardinality_columns]
df_encoded = pd.get_dummies(df, columns=low_cardinality_columns, drop_first=True)
```

**Why One-Hot Encoding**: 
- No ordinal relationship between categories (e.g., "software" is not "greater than" "biotech")
- `drop_first=True` avoids the **dummy variable trap** (perfect multicollinearity) — if you have k categories, k-1 dummies are sufficient because the kth is implied when all others are 0
- After generalization, cardinality is manageable (≤11 categories ≈ 10 new columns each)

#### 4.9.2 Label Encoding (for High-Cardinality Columns)

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

high_cardinality_columns = [col for col in df.select_dtypes(include=['object']).columns 
                             if df[col].nunique() > 100]

for col in high_cardinality_columns:
    df[col] = le.fit_transform(df[col].astype(str))
```

**Why Label Encoding for high cardinality**: One-hot encoding 100+ categories would create 100+ sparse columns. Label encoding assigns a single integer to each category, keeping dimensionality at 1 column. The tradeoff is that it introduces an artificial ordinal relationship, but for tree-based models this is acceptable.

**The >100 threshold**: Chosen as the boundary between "manageable for one-hot" and "too many columns". This is a pragmatic engineering decision.

#### 4.9.3 Status Target Variable Encoding

```python
# Binary encoding
df_encoded['status_label'] = df_encoded['status'].map({
    'acquired': 1, 'operating': 0
})

# Multi-class encoding (later)
df_encoded['status_label'] = df['status'].map({
    'acquired': 1, 'operating': 0, 'closed': 2, 'ipo': 3
})
```

**Why explicit mapping**: The target variable requires deterministic, meaningful encoding. Label auto-encoding could assign arbitrary numbers; explicit mapping ensures semantic consistency.

#### 4.9.4 Binary Encoding (Demonstrated in ML Pipeline Notebook)

```python
import category_encoders as ce
binary_encoder = ce.BinaryEncoder(cols=['Company', 'Department'])
df_binary_encoded = binary_encoder.fit_transform(df)
```

**Why Binary Encoding**: Encodes k categories into ⌈log₂(k)⌉ columns. For 100 categories, this creates ~7 columns instead of 100 (one-hot) or 1 (label). It's a middle ground: less dimensionality than one-hot, less information loss than label encoding.

#### 4.9.5 Frequency Encoding (Demonstrated)

```python
frequency_encoding = df['Department'].value_counts().to_dict()
df['Department_encoded'] = df['Department'].map(frequency_encoding)
```

**Why**: Replaces categories with their frequency count. Categories that appear often get higher values. Useful when the frequency of a category is correlated with the target.

#### 4.9.6 Target Encoding (Demonstrated)

```python
mean_salary = df.groupby('Department')['Salary'].mean()
df['Department_target_encoded'] = df['Department'].map(mean_salary)
```

**Why**: Replaces categories with the mean of the target variable for that category. Directly captures the category-target relationship. Risk of data leakage — requires careful cross-validation.

**Input**: DataFrame with categorical columns  
**Output**: DataFrame with all numerical columns  
**Dependencies**: `pandas`, `scikit-learn.LabelEncoder`, `category_encoders`

---

### 4.10 Correlation Analysis & Feature Selection

**Component Name**: Correlation-Based Feature Filter

**Functionality**: Computes Pearson correlation between all features and the target variable, then removes features with negligible correlation.

```python
# Encode target
df_encoded['status_label'] = df['status'].map({
    'acquired': 1, 'operating': 0, 'closed': 2, 'ipo': 3
})

# Compute correlation matrix
corr_matrix = df_encoded.corr()
corr = np.array(corr_matrix['status_label'])
cols = df_encoded.columns

# Identify low-correlation columns
columns_to_drop = []
for i in range(len(cols)):
    print(cols[i], "-->", np.round(corr[i], 3))
    if np.isnan(corr[i]) or np.abs(corr[i]) < 0.009:
        columns_to_drop.append(cols[i])

# Preserve essential columns
essential_columns = ['status_label', 'status_closed', 'status_ipo', 'status_operating']
columns_to_drop = [col for col in columns_to_drop if col not in essential_columns]

# Drop low-correlation features
df_encoded.drop(columns_to_drop, axis=1, inplace=True)

# Separate features and target
X = df_encoded.drop(columns=['status_label']).values
y = df_encoded['status_label'].values
```

**Why Pearson Correlation for Feature Selection**:
- Simple and interpretable — directly measures linear relationship strength
- Computationally cheap — O(n) per feature pair
- Works well as a first-pass filter before more expensive methods

**Why the 0.009 threshold**:
- This is an extremely low bar — essentially only removing features with near-zero correlation to the target
- Features with |r| < 0.009 have practically no linear predictive power
- This conservative threshold ensures we don't accidentally remove useful nonlinear predictors
- NaN correlations (from constant columns) are also dropped

**Why preserve essential columns**:
- `status_closed`, `status_ipo`, `status_operating` are one-hot encoded versions of the target — they have high correlation by definition and are needed for model training
- Dropping them would lose critical target information

**Input**: Fully encoded DataFrame  
**Output**: Reduced DataFrame with only correlated features + target  
**Dependencies**: `numpy`, `pandas`

---

### 4.11 Visualization & EDA

**Component Name**: Exploratory Data Analysis Suite

**Functionality**: Creates comprehensive visual representations of the data at various stages.

#### 4.11.1 Missing Data Heatmap
```python
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()
```
**Purpose**: Visualizes the pattern of missing data across all rows and columns. Yellow cells = missing. Reveals whether missingness is random or structured.  
**Why viridis**: Perceptually uniform colormap, accessible to colorblind viewers.

#### 4.11.2 Distribution Histograms
```python
# All numeric features at once
data.hist(figsize=(14, 12))

# Individual feature histograms with KDE
num_features = len(df.columns)
num_rows = (num_features // 4) + (1 if num_features % 4 != 0 else 0)
fig, axs = plt.subplots(num_rows, 4, figsize=(50, num_rows * 7.5))
for i in range(num_rows):
    for j in range(4):
        if count < num_features:
            sns.histplot(df[cols[count]], ax=axs[i, j], kde=True)
```
**Purpose**: Shows the distribution of each numerical feature. KDE overlay provides a smooth density estimate. Helps identify skewness, bimodality, or unusual patterns.

#### 4.11.3 Correlation Matrix Heatmap
```python
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
```
**Purpose**: Visualizes pairwise Pearson correlations. Red = positive, blue = negative. Annotated values show exact r. Helps identify multicollinearity and feature-target relationships.  
**Why coolwarm**: Diverging colormap centered at 0, intuitively showing positive vs negative correlations.

#### 4.11.4 Pairplot
```python
sns.pairplot(data)
```
**Purpose**: Scatterplots for all feature pairs + diagonal histograms. Reveals nonlinear relationships that correlation alone might miss.

#### 4.11.5 Box Plots
```python
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_encoded)
plt.xticks(rotation=90)

# Multiple numeric columns
sns.boxplot(data=data.select_dtypes(include=['number']))
```
**Purpose**: Shows distribution quartiles and outliers for each feature. Whiskers extend to 1.5×IQR; points beyond are outliers.

**Input**: DataFrames at various pipeline stages  
**Output**: Matplotlib/Seaborn plots (displayed inline in notebooks)  
**Dependencies**: `matplotlib`, `seaborn`

---

### 4.12 Data Labeling

**Component Name**: Custom Category Labeling System

**Functionality**: Creates hierarchical labels for companies based on their category codes and status.

```python
# Basic labeling function
def label_company_type(category_code):
    if category_code in ['web', 'network_hosting']:
        return 'Technology'
    elif category_code in ['games_video']:
        return 'Entertainment'
    else:
        return 'Other'

df['Company_Type'] = df['category_code'].apply(label_company_type)

# Advanced compound labeling
def label_company_status(Company_Type, category_code):
    if Company_Type == 'Entertainment':
        return 'Active - ' + label_company_type(category_code)
    elif Company_Type == 'Technology':
        return 'Acquired - ' + label_company_type(category_code)
    else:
        return 'Inactive - ' + label_company_type(category_code)

df['Company_Status_Label'] = df.apply(
    lambda row: label_company_status(row['Company_Type'], row['category_code']), axis=1
)
```

**Purpose**: Creates human-readable composite labels that combine industry type and predicted status. Useful for downstream analysis and reporting.

**Input**: DataFrame with `category_code` column  
**Output**: New columns `Company_Type` and `Company_Status_Label`  
**Dependencies**: `pandas`

---

### 4.13 Final Cleaned Dataset Export

**Component Name**: Data Persistence

**Functionality**: Saves the fully preprocessed dataset to CSV for downstream use.

```python
company.to_csv('cleaned_companies.csv', index=False)
```

**Why `index=False`**: Prevents pandas from writing the DataFrame index as an extra column. The cleaned data should have only meaningful columns.

**Input**: Fully cleaned, encoded, feature-engineered DataFrame  
**Output**: `cleaned_companies.csv` file  
**Dependencies**: `pandas`

---

## 5. Optimizers & Algorithms — Detailed Explanation

### 5.1 IQR-Based Outlier Detection Algorithm

**Algorithm Name**: Tukey's Fences (IQR Method)

**Mathematical Formula**:
```
IQR = Q3 - Q1
Lower Fence = Q1 - 1.5 × IQR
Upper Fence = Q3 + 1.5 × IQR
Outliers = values < Lower Fence OR values > Upper Fence
```

**How It Works**:
1. Sort all values of the feature
2. Calculate Q1 (25th percentile) — the value below which 25% of data falls
3. Calculate Q3 (75th percentile) — the value below which 75% of data falls
4. IQR = Q3 - Q1 — the range of the middle 50% of data
5. Any value more than 1.5×IQR below Q1 or above Q3 is an outlier
6. These outlier rows are removed from the dataset

**Parameters**:
| Parameter | Value | Effect |
|---|---|---|
| Multiplier | 1.5 | Standard Tukey fence. Increasing to 3.0 would only catch extreme outliers |
| Columns Applied | `funding_total_usd`, `funding_rounds` | Financial features most prone to extreme values |

**Why Chosen Over Other Methods**:
- **vs. Z-Score**: IQR doesn't assume normality — startup funding is heavily right-skewed
- **vs. Fixed Percentile Trimming**: IQR adapts to data shape; fixed percentiles might remove valid data
- **vs. DBSCAN/Isolation Forest**: Simpler, more interpretable, sufficient for univariate outlier detection

**Applied In**: `Startup_Steps_Data_Preprocessing.ipynb`

---

### 5.2 Z-Score Outlier Detection Algorithm

**Algorithm Name**: Standard Score (Z-Score) Method

**Mathematical Formula**:
```
Z = (X - μ) / σ
Outlier if |Z| > 3
```

**How It Works**:
1. Calculate the mean (μ) and standard deviation (σ) of the feature
2. For each value, compute how many standard deviations it is from the mean
3. Any value with |Z| > 3 is more than 3 standard deviations away — statistically extreme
4. These rows are filtered out

**Parameters**:
| Parameter | Value | Effect |
|---|---|---|
| Threshold | 3.0 | In a normal distribution, 99.7% of data falls within ±3σ |

**Why Chosen For Specific Columns**: 
- Works best when the underlying distribution is approximately normal
- Simple to implement with `scipy.stats.zscore`
- Complements IQR for features where normality approximately holds

**Applied In**: `ML_Pipeline_on_Startup_Acquisition.ipynb`

---

### 5.3 Correlation-Based Feature Selection Algorithm

**Algorithm Name**: Univariate Pearson Correlation Filter

**Mathematical Formula**:
```
r(X, Y) = Σ[(Xi - X̄)(Yi - Ȳ)] / √[Σ(Xi - X̄)² × Σ(Yi - Ȳ)²]

Feature is dropped if |r(feature, target)| < 0.009
```

**How It Works**:
1. Compute the full Pearson correlation matrix for all columns
2. Extract the correlation vector for the target column (`status_label`)
3. For each feature, check if its absolute correlation with the target is below 0.009
4. Also check for NaN correlations (from constant-value columns)
5. Mark qualifying features for removal
6. Exclude essential columns from removal (target-derived columns)
7. Drop marked features

**Parameters**:
| Parameter | Value | Effect |
|---|---|---|
| Threshold | 0.009 | Very conservative — only removes near-zero correlation features |
| Essential Columns | `status_label`, `status_closed`, `status_ipo`, `status_operating` | Protected from removal |

**Why This Threshold**:
- 0.009 is intentionally very low — it only removes features with essentially zero linear predictive power
- A higher threshold (e.g., 0.1) might discard features useful for nonlinear models
- This serves as a **pre-filter** to reduce noise, not as an aggressive feature selector

**Limitations**:
- Pearson correlation only detects **linear** relationships
- A feature with r=0 might still be a powerful nonlinear predictor
- Does not account for feature interactions

**Applied In**: `doing.ipynb`

---

### 5.4 Encoding Algorithms

#### 5.4.1 One-Hot Encoding (Primary)

**Algorithm**: Create a new binary column for each unique category value. Set value to 1 if the original row had that category, 0 otherwise.

**Formula**: For k categories, create k-1 columns (with `drop_first=True`)

**Why `drop_first=True`**: If a categorical column has values {A, B, C}, we only need columns for B and C. If both are 0, the value must be A. This avoids **perfect multicollinearity** which can cause issues with linear models and correlation calculations.

**Space Complexity**: O(n × k) where n = rows, k = unique categories

#### 5.4.2 Label Encoding

**Algorithm**: Map each unique category to a sequential integer (0, 1, 2, ..., k-1). Uses scikit-learn's `LabelEncoder.fit_transform()`.

**Space Complexity**: O(n × 1) — single column regardless of k

**Caveat**: Introduces an artificial ordinal relationship. "Category 3" is not inherently "greater than" "Category 1". This is acceptable for tree-based models (which split on thresholds) but problematic for linear models.

#### 5.4.3 Binary Encoding

**Algorithm**: 
1. Label encode categories (A→0, B→1, C→2...)
2. Convert integer to binary representation
3. Each binary digit becomes a separate column

**For k categories**: Creates ⌈log₂(k)⌉ columns

**Example**: 8 categories → 3 binary columns (000, 001, 010, ..., 111)

#### 5.4.4 Frequency Encoding

**Algorithm**: Replace each category with its count in the dataset.

**Formula**: `encoded(category) = count(category) / total_rows`

**Risk**: Different categories with the same frequency become indistinguishable.

#### 5.4.5 Target Encoding

**Algorithm**: Replace each category with the mean of the target variable for rows with that category.

**Formula**: `encoded(category) = mean(target | rows where column == category)`

**Risk**: Data leakage — the encoding uses target information. Must use proper cross-validation.

---

### 5.5 Chunking / Generalization Algorithm for High-Cardinality Categoricals

**Algorithm Name**: Frequency-Based Top-K Binning (Chunking)

**How It Works**:
```
1. Count frequency of each category: value_counts()
2. Sort by frequency (descending) — default behavior
3. Keep the Top-K categories (K=10 in this project)
4. Replace all categories ranked K+1 and below with 'other'
5. Apply one-hot encoding on the reduced set
```

**Parameters**:
| Parameter | Value | Rationale |
|---|---|---|
| K (top categories kept) | 10 | Captures majority of data distribution while keeping dimensionality ≤11 columns after encoding |
| Replacement value | `'other'` | Semantically clear — groups all infrequent categories as one |

**Why This Algorithm Was Used**:
- **Prevents curse of dimensionality**: 161 countries → 161 one-hot columns = very sparse, hard to train
- **Preserves signal**: Top-10 categories contain the vast majority of rows → most predictive power
- **Reduces overfitting**: Rare categories with few samples create noisy features
- **The 'other' bin preserves information**: Instead of dropping rare categories entirely, grouping them retains the information that a company belongs to a less common category

**Applied On**:
| Column | Original Cardinality | After Chunking | One-Hot Columns Created |
|---|---|---|---|
| `category_code` | 42 | 11 (10 + other) | 10 (drop_first) |
| `country_code` | 161 | 11 (10 + other) | 10 (drop_first) |

**Alternative Approaches Not Used**:
- **Embedding**: Would require neural network architecture — overkill for this dataset size
- **Target Encoding**: Risk of leakage without careful cross-validation
- **PCA on one-hot**: Extra complexity with unclear benefit

---

### 5.6 Null Value Threshold Filtering Algorithm

**Algorithm Name**: Column-Level Null Density Filter

**How It Works**:
```
1. For each column, calculate: null_percentage = column.isnull().mean() × 100
2. If null_percentage > threshold, mark column for removal
3. Exception list: columns needed for feature engineering (e.g., closed_at)
4. Drop marked columns
```

**Two Variants Used**:

| Notebook | Threshold | Logic | Effect |
|---|---|---|---|
| Preprocessing | 96% nulls | Drop columns with >96% nulls | Very permissive — only drops nearly empty columns |
| doing.ipynb | 45% data (=55% nulls) | `dropna(thresh=len(df)*0.45, axis=1)` | More aggressive — requires each column to have ≥45% non-null data |

**Why Two Different Thresholds**:
- The **96% threshold** (Preprocessing notebook) is intentionally conservative — it only removes columns that are almost entirely empty, preserving maximum information for the detailed cleaning steps that follow
- The **45% threshold** (doing.ipynb) is applied in a more exploratory context where aggressive cleanup simplifies analysis

**Why This Matters**: Imputing a column that is 97% null means 97% of the values are fabricated. This introduces massive noise and can create spurious patterns. The threshold ensures only columns with sufficient real data survive.

---

## 6. Data Types — Full Column Reference

The original dataset contains **44 columns**. Here is the complete reference:

| # | Column | Type | Used | Reason for Removal (if applicable) |
|---|---|---|---|---|
| 1 | `id` | int | No | Database ID — no predictive value |
| 2 | `Unnamed: 0.1` | int | No | Index artifact from CSV export |
| 3 | `entity_type` | str | No | Constant value "Company" |
| 4 | `entity_id` | int | No | Redundant identifier |
| 5 | `parent_id` | float | No | Mostly null, relationship indicator |
| 6 | `name` | str | No | Company name — identifier, not feature |
| 7 | `normalized_name` | str | No | Lowercase version of name |
| 8 | `permalink` | str | No | URL slug — irrelevant |
| 9 | `category_code` | str | **Yes** | Industry category (encoded) |
| 10 | `status` | str | **Yes** | **TARGET VARIABLE** |
| 11 | `founded_at` | date | **Yes** | Foundation date (converted to year) |
| 12 | `closed_at` | date | **Yes** | Closing date (used for age_in_days) |
| 13 | `domain` | str | No | Website domain — irrelevant |
| 14 | `homepage_url` | str | No | Website URL — irrelevant |
| 15 | `twitter_username` | str | No | Social media — irrelevant for this model |
| 16 | `logo_url` | str | No | Image URL — irrelevant |
| 17 | `logo_width` | float | No | Image metadata — irrelevant |
| 18 | `logo_height` | float | No | Image metadata — irrelevant |
| 19 | `short_description` | str | No | Text — would need NLP |
| 20 | `description` | str | No | Text — would need NLP |
| 21 | `overview` | str | No | Text — would need NLP |
| 22 | `tag_list` | str | No | Free-form tags — would need NLP |
| 23 | `country_code` | str | **Yes** | Country (encoded) |
| 24 | `state_code` | str | **No** | Dropped later (too granular for non-US) |
| 25 | `city` | str | No | Too granular |
| 26 | `region` | str | No | Too granular |
| 27 | `first_investment_at` | date | Varies | First investment date |
| 28 | `last_investment_at` | date | Varies | Last investment date |
| 29 | `investment_rounds` | int | Varies | Number of investment rounds |
| 30 | `invested_companies` | int | No | Nested data — irrelevant |
| 31 | `first_funding_at` | date | **Yes** | First funding date (→ year) |
| 32 | `last_funding_at` | date | **Yes** | Last funding date (→ year) |
| 33 | `funding_rounds` | int | **Yes** | Number of funding rounds |
| 34 | `funding_total_usd` | float | **Yes** | Total funding in USD |
| 35 | `first_milestone_at` | date | **Yes** | First milestone date (→ year) |
| 36 | `last_milestone_at` | date | **Yes** | Last milestone date (→ year) |
| 37 | `milestones` | int | **Yes** | Number of milestones |
| 38 | `relationships` | int | **Yes** | Number of relationships |
| 39 | `created_by` | str | No | Database metadata |
| 40 | `created_at` | date | No | Database metadata |
| 41 | `updated_at` | date | No | Database metadata |
| 42-44 | Various | Various | Varies | Dependent on null analysis |

---

## 7. Libraries & Dependencies

| Library | Version | Purpose | Specific Usage |
|---|---|---|---|
| `pandas` | ≥1.0 | Data manipulation | `read_csv`, `DataFrame`, `get_dummies`, `to_datetime`, `drop`, `fillna`, `dropna`, `drop_duplicates`, `corr`, `to_csv`, `apply`, `map`, `replace`, `value_counts`, `isnull`, `mean`, `quantile`, `select_dtypes`, `groupby` |
| `numpy` | ≥1.19 | Numerical computing | `np.abs`, `np.isnan`, `np.round`, `np.where`, `np.array` |
| `matplotlib` | ≥3.3 | Core plotting | `plt.figure`, `plt.subplots`, `plt.show`, `plt.tight_layout`, `plt.xticks`, `plt.title`, `plt.xlabel`, `plt.ylabel` |
| `seaborn` | ≥0.11 | Statistical visualization | `sns.heatmap`, `sns.histplot`, `sns.boxplot`, `sns.pairplot`, `sns.scatterplot`, `sns.set` |
| `scikit-learn` | ≥0.24 | ML preprocessing | `LabelEncoder.fit_transform` |
| `category_encoders` | ≥2.2 | Advanced encoding | `BinaryEncoder.fit_transform` |
| `scipy` | ≥1.5 | Statistics | `stats.zscore` |
| `warnings` | stdlib | Warning suppression | `warnings.filterwarnings('ignore')` |

---

## 8. Notebook-by-Notebook Detailed Walkthrough

### 8.1 `Startup_Steps_Data_Preprocessing.ipynb`

**Role**: Primary data preprocessing pipeline — takes raw data and produces a cleaned, encoded, feature-engineered dataset.

**Cell-by-Cell Breakdown**:

| Cell # | Operation | Code Summary |
|---|---|---|
| 1 | Markdown header | "Data Preprocessing Steps" |
| 2 | Import libraries | pandas, numpy, warnings, matplotlib, seaborn |
| 3 | Load data | `pd.read_csv("companies.csv")` — displays full DataFrame |
| 4 | Preview | `company.head()` — first 5 rows |
| 5 | Statistics | `company.describe()` — summary stats |
| 6 | Columns | `company.columns` — list all 44 columns |
| 7 | Markdown | Data cleaning plan with detailed steps |
| 8 | Markdown | Section 1.a header |
| 9 | Drop granular columns | Drop `region`, `city` |
| 10 | Markdown | Section 1.b header |
| 11 | Drop redundant columns | Drop `id`, `Unnamed: 0.1`, `entity_type`, etc. (8 columns) |
| 12 | Markdown | Section 1.c header |
| 13 | Drop irrelevant columns | Drop `domain`, `homepage_url`, etc. (14 columns) |
| 14 | Preview cleaned | `company.head()` |
| 15 | Markdown | Section 1.d header |
| 16 | Count duplicates | `company.duplicated().sum()` |
| 17 | Remove duplicates | `company.drop_duplicates(inplace=True)` |
| 18 | Verify dedup | `company.duplicated().sum()` — should be 0 |
| 19 | Markdown | Section 1.e header |
| 20 | View DataFrame | `company` — show shape |
| 21 | Null count | `company.isnull().sum()` |
| 22 | Drop high-null columns | Drop columns with >96% null (except `closed_at`) |
| 23 | Verify nulls | `company.isna().sum()` |
| 24 | Markdown | Section 2.a header |
| 25 | Drop critical null rows | Drop rows where `status`, `country_code`, `category_code`, `founded_at` are null |
| 26 | Verify | `company.isnull().sum()` |
| 27 | View shape | `company` |
| 28 | Markdown | Section 2.b header |
| 29 | Empty placeholder | (Outlier detection placeholder) |
| 30 | Markdown | IQR Summary |
| 31 | Markdown | Section 2.b.1 header |
| 32 | IQR calculation | Calculate IQR, bounds for `funding_total_usd` and `funding_rounds` |
| 33 | Markdown | Find outliers header |
| 34 | Find outliers | Identify outlier rows |
| 35 | Markdown | Drop outliers header |
| 36 | Drop outliers | Remove outlier rows from DataFrame |
| 37 | View result | `company` |
| 38 | Markdown | Contradictory data section |
| 39 | Comment | "Check contradictory data later" |
| 40 | Markdown | Date transformation overview |
| 41 | Markdown | Section 1.a (dates) header |
| 42 | Date → year conversion | Convert all 6 date columns to year integers |
| 43 | Markdown | Category generalization header |
| 44 | Category counts | `company['category_code'].value_counts()` |
| 45 | Category chunking | Keep top 10 categories, replace rest with 'other' |
| 46 | One-hot encode categories | `pd.get_dummies(company, columns=['category_code'], drop_first=True)` |
| 47 | Verify | (category_code already replaced by dummies) |
| 48 | Markdown | Country encoding header |
| 49 | Country counts | `company['country_code'].value_counts()` |
| 50 | Country chunking | Keep top 10 countries, replace rest with 'other' |
| 51 | One-hot encode countries | `pd.get_dummies(company, columns=['country_code'], drop_first=True)` |
| 52 | Verify | `company.head()` |
| 53-54 | Markdown | Feature engineering sections |
| 55 | Status encoding | Status value replacement (operating/ipo→1, closed/acquired→0) |
| 56 | isClosed feature | `company['isClosed'] = ...` |
| 57 | Verify | `company.head()` |
| 58 | Markdown | active_days section |
| 59 | Fill closed_at nulls | Operating companies get 2021, closed get 0 |
| 60-61 | Markdown | Subtask headers |
| 62 | Calculate age | `company['age_in_days'] = closed_at - founded_at` |
| 63 | Markdown | Drop closed_at header |
| 64 | Drop closed_at | `company.drop(['closed_at'], axis=1)` |
| 65-70 | Target variable work | (Placeholder cells) |
| 71 | View data | `company` |
| 72 | Null check | `company.isna().sum()` |
| 73 | Mean imputation | `company.fillna(company.mean(), inplace=True)` |
| 74 | Verify nulls | `company.isna().sum()` — should be 0 |
| 75 | Verify | `company.head()` |
| 76 | Drop more columns | Drop `state_code`, `funding_total_usd` |
| 77 | Shape check | `company.shape` |
| 78 | Final null check | `company.isna().sum()` |
| 79 | **SAVE** | `company.to_csv('cleaned_companies.csv', index=False)` |

---

### 8.2 `doing.ipynb`

**Role**: Feature engineering, encoding, correlation analysis, and visualization pipeline.

**Cell-by-Cell Breakdown**:

| Cell # | Operation | Code Summary |
|---|---|---|
| 1 | Import libraries | pandas, numpy, matplotlib |
| 2 | Load data | `pd.read_csv('companies.csv')` |
| 3 | Shape | `print(df.shape)` — (rows, 44) |
| 4 | Info | `df.info()` — dtypes, non-null counts |
| 5 | Preview | `df.head()` |
| 6 | Statistics | `df.describe()` |
| 7 | Null sum | `df.isna().sum()` |
| 8 | Sorted null sum | `df.isna().sum().sort_values(ascending=False)` |
| 9 | Null percentages | Print NaN % per column |
| 10 | **45% threshold drop** | Drop columns where <45% of rows have data |
| 11 | Recheck nulls | Print NaN % again |
| 12 | Count duplicates | `df.duplicated().sum()` |
| 13 | Remove duplicates | `df.drop_duplicates()` |
| 14 | Verify dedup | `df.duplicated().sum()` |
| 15 | Date parsing | Convert `last_milestone_at`, `first_milestone_at`, `founded_at` to datetime |
| 16 | Extract year | `df['founded_year'] = df['founded_at'].dt.year` |
| 17 | Shape check | `df.shape` |
| 18 | Drop columns | Remove `id`, `Unnamed: 0.1`, `name`, `permalink`, `homepage_url`, `logo_url`, `normalized_name` |
| 19 | Categorical fill | Fill NaN in object columns with 'Unknown' |
| 20 | Numerical fill | Fill NaN in number columns with 0 |
| 21 | Date re-parsing | Re-parse remaining date columns with `errors='coerce'` |
| 22 | NaN recheck | Print remaining NaN percentages |
| 23 | **Encoding** | LabelEncoder for high-cardinality (>100 unique), One-Hot for low cardinality |
| 24 | Status binary encode | `acquired→1, operating→0` |
| 25 | Preview encoded | `df_encoded.head()` |
| 26 | **Histograms** | Grid of histograms with KDE for all features |
| 27 | Data inspection | `df.head()`, `df.info()` |
| 28 | Correlation matrix | `df_encoded.corr()` |
| 29 | **Feature Selection** | Multiclass status encoding, correlation analysis, drop features with |r| < 0.009 |
| 30 | **Box plots** | Boxplot of all encoded features |

---

### 8.3 `ML_Pipeline_on_Startup_Acquisition.ipynb`

**Role**: Comprehensive EDA, multiple encoding technique demonstrations, and data labeling system.

**Cell-by-Cell Breakdown**:

| Cell # | Operation | Code Summary |
|---|---|---|
| 1 | Import libraries | numpy, pandas, seaborn |
| 2 | Load data | `pd.read_csv('/content/companies.csv')` (Colab path) |
| 3 | Markdown | "Explore the data set" |
| 4 | Info | `data.info()` |
| 5 | Preview | `data.head()` |
| 6 | Statistics | `data.describe()` |
| 7 | Markdown | "Identify missing values" |
| 8 | Null count | `data.isnull().sum()` |
| 9 | **Missing data heatmap** | Viridis heatmap of null pattern |
| 10 | Markdown | "Distribution" |
| 11 | Histograms | `data.hist(figsize=(14, 12))` |
| 12 | Column names | `print(data.columns)` |
| 13 | Value counts | `data['twitter_username'].value_counts()` |
| 14 | Markdown | "Explore Relationships" |
| 15 | Data types | `data.dtypes` |
| 16 | Correlation matrix | Filter numeric, compute correlations |
| 17 | **Correlation heatmap** | Annotated coolwarm heatmap |
| 18 | **Pairplot** | `sns.pairplot(data)` |
| 19 | Scatterplot | `sns.scatterplot(data)` |
| 20 | Markdown | "Data Imbalances" |
| 21 | Value counts | Twitter username frequency |
| 22 | Group analysis | Mean values grouped by twitter_username |
| 23-24 | Markdown | "Outlier Detection" |
| 25 | Single column boxplot | `sns.boxplot(x=data['twitter_username'])` |
| 26 | Markdown | "Multiple Columns" |
| 27 | Multi-column boxplots | Boxplots for all numeric columns |
| 28 | Column list | `data.columns` |
| 29 | Grouped boxplot | `twitter_username` vs `city` |
| 30 | Markdown | "Data Cleaning" section |
| 31 | Null recheck | `data.isnull().sum()` |
| 32 | Markdown | "Remove rows/columns" |
| 33 | Drop nulls | `data.dropna(axis=0)`, `data.dropna(axis=1)` |
| 34 | Markdown | "Impute" |
| 35 | Imputation | Mean, median, 'Unknown' fill |
| 36 | Replace errors | `data['col'].replace('errored_value', 'correct_value')` |
| 37 | Dedup | `data.drop_duplicates()` |
| 38 | Z-score outliers | `stats.zscore` with threshold 3 |
| 39 | IQR capping | Cap values at IQR boundaries |
| 40 | Drop columns | `data.drop(['region'])` |
| 41 | Type conversion | Cast to float/category |
| 42 | String cleaning function | Strip, lowercase, convert to string |
| 43 | Info | `data.info()` |
| 44 | Describe | `data.describe()` |
| 45 | Markdown | "Data Encoding" |
| 46 | Company names | `data['name'].unique()` |
| 47 | **Encoding demo** | Full demo: Label, One-Hot, Binary, Frequency, Target encoding on sample data |
| 48 | Markdown | "Data Labelling" |
| 49 | Preview | `data.head()` |
| 50 | Markdown | "Apply Labeling Function" |
| 51 | **Labeling system** | `label_company_type()`, `label_company_status()` functions |
| 52 | Markdown | "Handling Missing Data" |
| 53 | Handle nulls before labeling | Fill NaN with 'Unknown', re-apply labeling |
| 54 | Preview | `df.head()` |
| 55 | **Advanced labeling** | Compound status + type labels |
| 56 | Empty | End of notebook |

---

## 9. Why Each Decision Was Made

| Decision | Rationale |
|---|---|
| **Drop region/city** | Too granular — thousands of unique values would not generalize. Country-level is sufficient for geographic signal. |
| **Drop id/entity columns** | Database artifacts with zero predictive value. Including them would cause overfitting to arbitrary IDs. |
| **Drop text/URL columns** | Would require NLP/text processing. This is a tabular ML pipeline, not an NLP pipeline. |
| **96% null threshold** | Columns nearly entirely empty. Imputing 96% of values = fabricating data. |
| **45% null threshold** | Stricter alternative — ensures every feature has substantial real data. |
| **Drop rows for status/country/category/founded_at** | Critical features that cannot be meaningfully imputed. A fake status value would corrupt the target. |
| **IQR for funding outliers** | Funding data is right-skewed (few mega-funded companies). IQR doesn't assume normality, unlike Z-score. |
| **Z-score for other outliers** | When distribution is approximately normal, Z-score is simpler and well-understood. |
| **Dates → years** | Full dates add noise; year captures the meaningful temporal signal (era, market conditions). |
| **Top-10 chunking for categories** | Prevents dimensionality explosion. Top-10 captures Pareto majority. 'other' preserves minority signal. |
| **One-Hot for low cardinality** | No ordinal assumption. Correct for nominal categories. `drop_first=True` avoids multicollinearity. |
| **Label encoding for high cardinality** | One-hot would create too many sparse columns (>100). Label encoding keeps 1 column. |
| **Correlation threshold 0.009** | Conservative filter — only removes features with essentially zero linear correlation. Preserves potential nonlinear predictors. |
| **Mean imputation for remaining nulls** | Simple, preserves distribution center. Better than 0 for features where 0 is not a natural value. |
| **'Unknown' for categorical nulls** | Preserves missingness information as a category. Better than mode imputation which hides missing data patterns. |
| **isClosed binary feature** | Simplifies multi-class target into binary for certain analyses. Captures a key business outcome. |
| **age_in_days feature** | Captures company longevity — a strong predictor of success. Operating companies get 2021 (current year at data collection). |
| **Save to cleaned_companies.csv** | Separates preprocessing from modeling. Allows reuse without re-running the entire pipeline. |

---

## 10. Summary

This project implements a **complete end-to-end machine learning data pipeline** for predicting startup acquisition status. The pipeline spans three Jupyter notebooks:

1. **`Startup_Steps_Data_Preprocessing.ipynb`** — The core preprocessing engine: column pruning (24+ columns removed), deduplication, null handling (96% threshold + critical-row dropping), IQR-based outlier removal, date-to-year transformation, frequency-based categorical chunking (top-10 + 'other'), one-hot encoding, feature engineering (isClosed, age_in_days), mean imputation, and export to `cleaned_companies.csv`.

2. **`doing.ipynb`** — Feature engineering and analysis: 45% null threshold filtering, LabelEncoder for high-cardinality (>100 unique) and one-hot for low-cardinality columns, multi-class status encoding (acquired=1, operating=0, closed=2, ipo=3), Pearson correlation-based feature selection (drop |r| < 0.009), and comprehensive visualizations (histograms with KDE, boxplots).

3. **`ML_Pipeline_on_Startup_Acquisition.ipynb`** — Full EDA and encoding reference: missing data heatmap (viridis), correlation heatmap (coolwarm, annotated), pairplots, scatterplots, boxplots (single/multi/grouped), Z-score and IQR outlier methods, five encoding techniques demonstrated (Label, One-Hot, Binary, Frequency, Target), and a custom data labeling system mapping categories to company types and status labels.

**Key Algorithms**: IQR outlier detection (Tukey's fences, 1.5× multiplier), Z-score filtering (|Z| > 3), Pearson correlation feature selection (0.009 threshold), frequency-based top-K categorical binning (K=10), one-hot encoding with `drop_first=True`, and LabelEncoder for high-cardinality features.

**Dataset**: 44 columns, ~196K+ rows of CrunchBase startup data. After preprocessing: ~20-25 encoded feature columns ready for model training. Target variable: `status` (4 classes: operating, ipo, acquired, closed).
