🧠 Model Setup & Project Run Guide
Step 1: Join the Model Files

The model is split into multiple parts inside the models/ folder.
Run the following command in your terminal to combine them into one file:

🪄 For Windows:
**copy /b models\best_linear_svm_alexnet_car.part* models\best_linear_svm_alexnet_car.pth
**
🪄 For Mac/Linux:
**cat models/best_linear_svm_alexnet_car.part* > models/best_linear_svm_alexnet_car.pth
**
Step 2: Install Required Libraries
**pip install -r requirements.txt
**
Step 3: Run the Application
**streamlit run ui_code.py
**

Then open the URL shown in the terminal (usually http://localhost:8501) in your browser.


