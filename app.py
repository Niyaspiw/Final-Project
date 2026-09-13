import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/wine_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Convert feature names: replace spaces with underscores to match HTML form
feature_names = [str(f).strip().replace(' ', '_') for f in joblib.load('model/features.pkl')]
print("Loaded features:", feature_names)


def generate_wine_profile(data):
    """Generate trait pills and a descriptive paragraph from input values."""
    traits = []

    alcohol = float(data['alcohol'])
    volatile_acidity = float(data['volatile_acidity'])
    total_so2 = float(data['total_sulfur_dioxide'])
    residual_sugar = float(data['residual_sugar'])
    density = float(data['density'])
    sulphates = float(data['sulphates'])

    # --- Trait pills (unchanged) ---
    if alcohol >= 12:
        traits.append(("🔥", "High Alcohol"))
    elif alcohol < 10:
        traits.append(("💧", "Low Alcohol"))

    if volatile_acidity >= 0.7:
        traits.append(("🍋", "High Acidity"))
    elif volatile_acidity < 0.3:
        traits.append(("⚖️", "Balanced Acidity"))

    if total_so2 >= 100:
        traits.append(("🧂", "High Sulfites"))
    elif total_so2 < 50:
        traits.append(("🌿", "Low Sulfites"))

    if residual_sugar >= 6:
        traits.append(("🍯", "Sweet"))
    elif residual_sugar < 3:
        traits.append(("🥂", "Dry"))

    if alcohol >= 12 and density >= 0.996:
        traits.append(("💪", "Full Body"))
    elif alcohol < 10.5:
        traits.append(("🪶", "Light Body"))

    if sulphates >= 0.7:
        traits.append(("⏳", "Good Aging Potential"))
    if volatile_acidity >= 0.8:
        traits.append(("⚠️", "Drink Soon"))

    # --- Body sentence (part 1) ---
    if alcohol >= 12:
        body = "This is a bold, full-bodied wine with a rich, warming character."
    elif alcohol < 10:
        body = "This is a light, easy-drinking wine with a delicate character."
    else:
        body = "This is a well-balanced, medium-bodied wine."

    # --- Flavor sentence (part 2) ---
    flavor_parts = []

    if residual_sugar >= 6:
        flavor_parts.append("a sweet, honeyed finish")
    elif residual_sugar < 3:
        flavor_parts.append("a crisp, dry finish")

    if volatile_acidity >= 0.7:
        flavor_parts.append("a sharp acidic edge on the palate")
    elif volatile_acidity < 0.3:
        flavor_parts.append("smooth, well-rounded acidity")

    if total_so2 >= 100:
        flavor_parts.append("high sulfite levels suited for aging")
    elif total_so2 < 50:
        flavor_parts.append("low sulfites meant for early enjoyment")

    flavor = "On the palate, it offers " + ", ".join(flavor_parts[:-1])
    if len(flavor_parts) > 1:
        flavor += ", and " + flavor_parts[-1]
    else:
        flavor = "On the palate, it offers " + flavor_parts[0]
    flavor += "."

    # --- Closing note (part 3) ---
    closing = ""
    if sulphates >= 0.7 and volatile_acidity < 0.8:
        closing = " With strong aging potential, this wine will reward patience."
    elif volatile_acidity >= 0.8:
        closing = " Best enjoyed soon while its acidity is still lively."

    paragraph = f"{body} {flavor}{closing}"

    return traits, paragraph


@app.route('/')
def home():
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Form data received:", dict(request.form))

        values = []
        for feat in feature_names:
            if feat not in request.form:
                raise KeyError(f"Form is missing: {feat}")
            values.append(float(request.form[feat]))

        input_array = np.array([values])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            result = "Good Wine"
            confidence = round(probability[1] * 100, 2)
            color = "green"
        else:
            result = "Bad Wine"
            confidence = round(probability[0] * 100, 2)
            color = "red"

        traits, paragraph = generate_wine_profile(request.form)

        return render_template(
            'index.html',
            features=feature_names,
            prediction_text=result,
            confidence=f"{confidence}%",
            color=color,
            traits=traits,
            paragraph=paragraph,
            values=request.form
        )

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return render_template(
            'index.html',
            features=feature_names,
            prediction_text=f"Error: {e}",
            confidence="0%",
            color="red",
            values=request.form
        ), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)