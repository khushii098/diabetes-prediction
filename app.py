from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os
from flask import Flask, render_template, request, jsonify

import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key="AIzaSyD75sovJC9w55n5tnlwkF2vkpAz__bVY7E")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[
    {"role": "user", "parts": ["You're an AI health assistant.Don't give any disclamer just give to the point answers to only to questions realted "
    "to medical problems else say the question is not relevant. Help users with diabetes-related questions."]}
])



@app.route("/chat", methods=["POST"])
def chat_response():
    data = request.get_json()
    user_input = data.get("message", "")
    try:
        response = chat.send_message(user_input)
        return jsonify({"reply": response.text})
    except Exception as e:
        print("Gemini Error:", e)
        return jsonify({"reply": f"❌ Error: {str(e)}"})

app.secret_key = 'your_secret_key'  # Needed for session

# Load model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/gender', methods=['GET', 'POST'])
def gender():
    if request.method == 'POST':
        session['gender'] = request.form['gender']
        return redirect(url_for('form'))
    return render_template('gender.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    gender = session.get('gender', None)

    if request.method == 'POST':
        pregnancies = int(request.form.get('pregnancies', 0)) if gender == 'female' else 0
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
        dpf = float(request.form['diabetespedigreefunction'])
        #

        # Calculate Diabetes Pedigree Function
        #dpf = diabetic_family / total_family if total_family > 0 else 0.0

        # Model input
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]

        # Store prediction in session
        session['prediction'] = int (prediction)
        probability = model.predict_proba(input_data)[0][1] * 100
        session['prediction'] = int(prediction)
        session['probability'] = round(probability, 2)


        return redirect(url_for('result'))

    return render_template('form.html', gender=gender)

@app.route('/result')
def result():
    prediction = session.get('prediction', None)
    diabetes_percentage = session.get('probability', None)

    if prediction == 0:
        risk = "less chance of getting Diabetes"
        suggestions = [
            "You can maintain a healthy balanced diet by prioritzing whole grains, fresh vegetables, and lean proteins.",
            "Aim for at least 30 minutes of moderate exercise, such as brisk walking, cycling, or swimming, most days of the week.",
            "Even small weight loss can significantly reduce diabetes risk, especially for those with prediabetes.",
            "Practice relaxation techniques like meditation, yoga, or deep breathing to help regulate blood sugar levels",
            "Avoid smoking & limit alcohol, Prioritize sleep, and do regular health check-ups."
        ]
        videos = [
    {
        "title": "10 Minute Exercise For Diabetes (LOW IMPACT!)",
        "url": "https://www.youtube.com/watch?v=NbYTSPqq1R4"
    },
    {
        "title": "Healthy Eating with Diabetes",
        "url": "https://www.youtube.com/watch?v=wOIZEz0hAY4"
    }
    ]

        show_map = False
    elif prediction ==1:
        risk="there may be mild diabetes" 
        suggestions = [
            "Incorporate curry leaves into your diet, curry leaves help lower blood glucose levels and improve insulin activity. Chewing 10–15 fresh leaves daily can be beneficial..",
            "Consume jamun seed powder daily:- Jamun seed powder contains jamboline, which helps convert starch into energy and regulates blood sugar levels. Consume half a teaspoon of powdered seeds with water daily..",
            "Soak and drink fenugreek seeds water:- Soak 1–2 teaspoons of fenugreek seeds in water overnight and drink the water on an empty stomach. It helps enhance glucose tolerance and lower blood sugar levels..",
            "Include amla juice in your morning routine:- Amla is high in Vitamin C and antioxidants. Mix 1–2 tablespoons of amla juice with water and drink daily to regulate pancreas function and insulin levels..",
            "Try okra water for blood sugar control:- Rich in fiber and antioxidants, okra helps stabilize blood sugar. Soak sliced okra in water overnight and drink the water in the morning..",
            "Use cinnamon in your meals:- This spice helps increase insulin sensitivity. Add 1/2 teaspoon daily to tea or food.."
        ]
        videos = [
            {"title": "Diabetes Exercises For Type 2 Diabetes Workout At Home", "url": "https://www.youtube.com/watch?v=6MnbaUBO_DY"},
            {"title": "Healthy Eating with Diabetes", "url": "https://www.youtube.com/watch?v=wOIZEz0hAY4"}
        ]
        show_map = False   
    else:
        risk = "High Diabetes"
        suggestions = [
            "Consult a healthcare professional immediately.",
            "Monitor your blood sugar levels regularly.",
            "Follow a strict diabetes-friendly diet.",
            "Engage in regular physical activity.",
            "Stay hydrated and avoid sugary drinks.",
            "Manage stress through relaxation techniques."
        ]
        videos = [
            {"title": "10 Best Diabetes Exercises to Lower Blood Sugar", "url": "https://www.youtube.com/watch?v=-uK8a80vyeI"},
            {"title": "Managing Diabetes: Diet Changes", "url": "https://www.youtube.com/watch?v=4o2-M1C6T5I"}
        ]
        show_map = True

    return render_template('result.html',
                           risk=risk,
                           suggestions=suggestions,
                           videos=videos,
                           show_map=show_map,
                           diabetes_percentage=diabetes_percentage)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # use PORT env var on Render
    app.run(host='0.0.0.0', port=port, debug=True)
