import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
import pickle

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open("logistic.pkl", "rb"))

# HTML Template with Tailwind CSS for modern UI design
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-950 min-text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-3xl w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl p-8 my-8 text-slate-100">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
                Loan Eligibility Predictor
            </h1>
            <p class="text-slate-400 text-sm mt-2">Fill in the applicant details below to check loan approval status.</p>
        </div>

        <!-- Prediction Result Card -->
        {% if prediction_text %}
        <div class="mb-8 p-4 rounded-xl text-center font-medium border transition-all duration-300 {{ 'bg-emerald-950/50 border-emerald-500/50 text-emerald-300' if 'Approved' in prediction_text else 'bg-rose-950/50 border-rose-500/50 text-rose-300' }}">
            <p class="text-lg">{{ prediction_text }}</p>
        </div>
        {% endif %}

        <!-- Form -->
        <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Gender -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Gender</label>
                <select name="Gender" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <!-- Married -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Marital Status</label>
                <select name="Married" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <!-- Dependents -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Dependents</label>
                <select name="Dependents" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3+</option>
                </select>
            </div>

            <!-- Education -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Education</label>
                <select name="Education" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="1">Graduate</option>
                    <option value="0">Not Graduate</option>
                </select>
            </div>

            <!-- Self Employed -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Self Employed</label>
                <select name="Self_Employed" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <!-- Credit History -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Credit History</label>
                <select name="Credit_History" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="1">Yes (Good History)</option>
                    <option value="0">No (Bad History)</option>
                </select>
            </div>

            <!-- Applicant Income -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Applicant Income (₹)</label>
                <input type="number" name="ApplicantIncome" placeholder="e.g. 5000" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
            </div>

            <!-- Coapplicant Income -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Coapplicant Income (₹)</label>
                <input type="number" name="CoapplicantIncome" placeholder="e.g. 2000" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
            </div>

            <!-- Loan Amount -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Loan Amount (in thousands)</label>
                <input type="number" name="LoanAmount" placeholder="e.g. 150" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
            </div>

            <!-- Loan Amount Term -->
            <div>
                <label class="block text-sm font-medium text-slate-300 mb-2">Loan Term (Months)</label>
                <input type="number" name="Loan_Amount_Term" placeholder="e.g. 360" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
            </div>

            <!-- Property Area -->
            <div class="md:col-span-2">
                <label class="block text-sm font-medium text-slate-300 mb-2">Property Area</label>
                <select name="Property_Area" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500" required>
                    <option value="2">Urban</option>
                    <option value="1">Semiurban</option>
                    <option value="0">Rural</option>
                </select>
            </div>

            <!-- Submit Button -->
            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold py-3 rounded-xl shadow-lg transition-all duration-200 cursor-pointer">
                    Predict Loan Status
                </button>
            </div>

        </form>
    </div>
</body>
</html>
"""


@app.route("/")
def home():
  return render_template_string(TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
  try:
    # Extract form inputs matching features in logistic.pkl
    features = [
        int(request.form["Gender"]),
        int(request.form["Married"]),
        int(request.form["Dependents"]),
        int(request.form["Education"]),
        int(request.form["Self_Employed"]),
        float(request.form["ApplicantIncome"]),
        float(request.form["CoapplicantIncome"]),
        float(request.form["LoanAmount"]),
        float(request.form["Loan_Amount_Term"]),
        float(request.form["Credit_History"]),
        int(request.form["Property_Area"]),
    ]

    # Reshape features into a 2D array for the model
    final_features = np.array([features])
    prediction = model.predict(final_features)

    # Assuming 1 = Approved, 0 = Rejected
    result_text = (
        "Congratulations! Your Loan is Approved."
        if prediction[0] == 1
        else "Sorry, your Loan Application is Rejected."
    )

    return render_template_string(TEMPLATE, prediction_text=result_text)

  except Exception as e:
    return render_template_string(
        TEMPLATE, prediction_text=f"Error in processing input: {str(e)}"
    )


if __name__ == "__main__":
  app.run(debug=True)
