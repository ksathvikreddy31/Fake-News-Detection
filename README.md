# 📰 Fake News Detection

## 📖 Description
A machine learning-based **Fake News Detection** system built using **Python** and **Natural Language Processing (NLP)**. This project aims to classify news articles as **fake** or **real** using a trained model.

## 🚀 Features
- Uses **TF-IDF Vectorization** for text processing.
- **Machine Learning Model** trained for classification.
- Web-based interface for easy news validation.
- Lightweight and efficient detection system.

## 🛠️ Technologies Used
- **Python** – Core programming language.
- **Flask** – Web framework for serving the application.
- **Scikit-Learn** – Machine learning library.
- **Pandas & NumPy** – Data processing.
- **Pickle** – Model serialization.
- **HTML & CSS** – Frontend interface.

## 📂 Project Structure
```
📁 Fake-News-Detection
 ├── 📁 static        # Contains CSS files
 ├── 📁 templates     # HTML files for UI
 ├── 📁 News_dataset  # Dataset used for training
 ├── 📜 app.py       # Flask application
 ├── 📜 main.py      # Core script for ML model
 ├── 📜 model.pkl    # Trained ML model
 ├── 📜 vectorizer.pkl # TF-IDF Vectorizer
 ├── 📜 README.md    # Project documentation
```

## 📥 Installation & Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ksathvikreddy31/Fake-News-Detection.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd Fake-News-Detection
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask app**:
   ```bash
   python app.py
   ```
5. **Access the web interface**:  
   Open `http://127.0.0.1:5000/` in your browser.

## 🎯 Future Improvements
- Enhance the model accuracy.
- Add API integration for real-time news validation.
- Improve UI/UX for better user interaction.

## ✅ Conclusion
This project demonstrates **Machine Learning, NLP, and Web Development**. It helps users differentiate between **fake** and **real** news efficiently.
