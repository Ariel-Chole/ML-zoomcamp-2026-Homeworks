# Load the model
import pickle

output_file= 'model_pipeline.bin'

with open (output_file,"rb") as f_in:
    model = pickle.load(f_in)

customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

def predict(customer):
    predict_proba = model.predict_proba(customer)[0,1]
    converted = predict_proba >= 0.5 
    result = {
        "converted probability": float(predict_proba),
        "converted" : bool(converted)
    }
    return result


convert_prediction = predict(customer)

print("input", customer)
print("converted probability",convert_prediction)