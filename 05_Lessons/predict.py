
import pickle
# Load the model
output_file = 'model_C = 1.0.bin'

with open(output_file,"rb") as f_in:
   (model,dv)=pickle.load(f_in)

student = {'lead_source': 'paid_ads',
 'industry': 'retail',
 'employment_status': 'employed',
 'location': 'asia',
 'annual_income': 25824.0,
 'number_of_courses_viewed': 1,
 'interaction_count': 4,
 'lead_score': 0.41}

#handling categorical features
X = dv.transform(student)
X
#predicting new student converted probability
predict_prob = model.predict_proba(X)[0,1]
print("input", student)
print("converted probability",predict_prob)

if predict_prob >=0.5:
   print("Customer will purchase the course.")
else:
   print("Customer will not purchase the course.")