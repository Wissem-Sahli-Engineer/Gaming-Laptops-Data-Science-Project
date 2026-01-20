# 🎮 Gaming Laptops Data Science Project (Tunisia)

## 📌 Project Overview 

This **Data Science project** aims to help **gamers, students, and general users in Tunisia** choose the **best price / quality gaming laptop**.
### website : https://www.tunisianet.com.tn/

The project follows the **full data science pipeline**:
- Data collection (web scraping)
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Machine learning model (future step)
- Dashboard & deployment

All steps are implemented using **Python and its data science libraries**.

---

## 🧰 Technologies & Libraries Used

- Python  
- Pandas, NumPy  
- Selenium (Web Scraping)  
- Matplotlib, Seaborn  
- Plotly & Dash  

---

## ⚙️ How to Run the Project

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## 📂 Project Structure
````text
DS-project/
│
├── 1scrap.py # Data collection (web scraping)
├── 2cleaning.py # Data cleaning & preprocessing
├── 3plotting.py # Visualization functions
├── 4EDA.py # Exploratory Data Analysis
├── app.py # Dash dashboard
├── gaming_laptops.csv # df using first scraping method-(Title Split)
├── gaming_laptops2.csv # df using second scraping method-(Description Parsing)
├── gaming_laptops2_cleaned.csv
�   links.txt # HTML code ! 
├── requirements.txt
└── README.md
````

---

## 🕷️ 1. Data Collection (Scrapping.py)

Data was collected using **Selenium** because the source website is **dynamic**, so traditional scraping tools like *BeautifulSoup* were not suitable.

### Key Points:
- Selenium was used to navigate pages and handle pagination
- Website structure was inspected to identify required HTML classes
- Pandas was used to store the data in a DataFrame
- The final dataset was saved as a CSV file

### Scraping Methods:
Two scraping approaches were used:
1. **Title-based extraction** (string splitting)
2. **Description parsing**

After comparison, the **title-based method contained more missing values**, so the description-based method was not used.

---

## 🧹 2. Data Cleaning & Preprocessing

Data cleaning was performed to improve data quality and consistency.

### Cleaning Steps:
- Used the first DataFrame to fill missing values in the second DataFrame
- Converted **Price** and **Warranty (Garentie)** to correct numeric formats
- Removed duplicate records
- Fixed GPU naming inconsistencies  
  Example: `RTX 3050 4GB` and `RTX 3050` were treated as the same GPU
- Filled missing categorical values:
  - `GPU` and `Color` were filled with `"Not Listed"`  
  - These represented less than **1%** of the data

### Notes:
- The **Refresh Rate** feature had many missing values
- No imputation was done because most screens had **144Hz or higher**
- This feature was not used in further analysis

---

## 📊 3. Exploratory Data Analysis (EDA)

EDA was conducted using:
- Matplotlib
- Seaborn
- Pandas built-in functions

### Visualizations:
- Price distribution
- Price vs GPU
- Price vs Storage
- RAM distribution
- Correlation heatmap
- Availability distribution

### Plotting Strategy:
Two visualization approaches were implemented in `plotting.py`:
1. `individual_plots()` – single charts
2. `subplots()` – grouped dashboard-style figures

These functions were later used in `EDA.py`.

Additionally, Pandas functions such as:
- `head()`
- `describe()`
- `value_counts()`

were used for quick statistical insights.

---

## 📈 4. Dashboard (Dash)

An interactive dashboard was built using **Dash and Plotly** to:
- Visualize key insights in one place
- Improve comparison between features
- Prepare the project for deployment

---

## 🔮 Future Work

- Build a price prediction model
- Add feature importance analysis
- Improve dashboard with filters (GPU, RAM, Price range)
- Deploy the app online

---

## 👤 Author

**Wissem Sahli**  




















