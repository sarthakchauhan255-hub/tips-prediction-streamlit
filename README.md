# 🍽️ Tips Prediction ML Web App

A **Machine Learning web application** that predicts the **tip amount** based on the total restaurant bill using a **Linear Regression model**.

The application is built using **Python, Scikit-Learn, and Streamlit**, providing an interactive user interface where users can input bill information and instantly receive a predicted tip.
## 🌐 Live Demo
https://tips-prediction-app-mgghwbanbqm69omeabraju.streamlit.app

---

## 📌 Project Overview

In restaurants, tipping behavior often depends on the total bill amount and other factors.  
This project builds a **Linear Regression model** to learn the relationship between the bill amount and the tip.

The model is trained using the **Seaborn Tips dataset**, and the trained model is deployed using **Streamlit** to create an interactive web application.

---
## How It Works

1. User inputs restaurant bill details
2. Data is passed to a trained Linear Regression model
3. Model predicts expected tip amount
4. Streamlit UI displays prediction and analytics

## 🚀 Features

- Interactive **Streamlit user interface**
- Real-time **tip prediction**
- Machine Learning model built with **Scikit-Learn**
- Clean and simple web application
- Ready for **cloud deployment**

---

## 🧠 Machine Learning Model

**Algorithm Used**

- Linear Regression

**Libraries Used**

- Scikit-learn
- Pandas
- NumPy

**Dataset**

- Seaborn Tips Dataset

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

```
ml-deploy/
│
├── app.py
├── tips_model.pkl
├── requirements.txt
├── README.md
└── tips_training.ipynb
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sarthakchauhan255-hub/tips-prediction-ml.git
cd tips-prediction-ml
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate it (Windows):

```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
streamlit run app.py
```

The application will open in your browser:

```
http://localhost:8501
```

---

## 🌐 Live Demo

After deployment on **Streamlit Cloud**, the app will be available at:

```
https://tips-prediction-app-mgghwbanbqm69omeabraju.streamlit.app
```

---
## Model Performance

MAE: 0.73  
RMSE: 1.02  
R² Score: 0.56

## 📊 Example Prediction

Input:

```
Total Bill: 25
```

Output:

```
Predicted Tip: $4.02
```

---


## 👨‍💻 Author

**Sarthak Chauhan**

Computer Science Student | Machine Learning Enthusiast

---

## ⭐ Future Improvements

- Add more input features (size, day, smoker, etc.)
- Improve UI design
- Use more advanced regression models
- Add model evaluation metrics in the app

---

## 📜 License

This project is open-source and available under the MIT License.
