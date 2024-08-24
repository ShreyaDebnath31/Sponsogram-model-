import streamlit as st
import pickle
import numpy as np
import sys
print(f"Python executable: {sys.executable}")

# Load the saved model
with open('light_gbm2.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")
print(type(model))

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded successfully!")

# Set up the Streamlit interface
st.title('ROI Prediction')

st.header('Enter the campaign data:')

duration = st.number_input('Campaign Duration', value=0)
budget = st.number_input('Campaign Budget(in rupees)', value=0.0)

st.header('Enter the influencer data:')

followers = st.number_input('Number of followers of the influencer', value=0.0)
eng_rate = st.number_input('Engagement Rate of the influencer', value=0.0)
avg_likes = st.number_input('Average likes on influencer posts', value=0.0)
avg_comm = st.number_input('Average comments on influencer posts', value=0.0)

totalMetrics = followers*eng_rate*budget/100000
engagementMetrics = duration * avg_likes * avg_comm / 100000

# input_df = {
#     'influencer_followers': data['Influencer_Followers'],
#     'Total_Metrics' : [(data['Engagement_Rate'][0] * data['Amount_Spent'][0] * data['Influencer_Followers'][0] )/100000],
#     'Engagement_Metrics' : (  data['Duration_Days'][0] * data['Avg_Likes'][0] * data['Avg_Comments'][0] / 100000)
#   }

# Prediction button
if st.button('Predict'):
    # Create a numpy array of the input features
    input_features = np.array([[followers, totalMetrics, engagementMetrics]])
    
    # Make the prediction
    prediction = model.predict(input_features)

    roi = (prediction-budget)/budget
    # transformed_roi = np.array([roi])

    # transformed_roi = scaler.transform(transformed_roi)
        
        # Display the prediction
    # st.write(f'prediction {prediction}')
    # st.write(f'roi {roi}')
    # st.write(f'Transformed roi {transformed_roi}')


    #  (copy_x_test['ROI'] < 0.6),
    # (copy_x_test['ROI'] >= 0.6) & (copy_x_test['ROI'] < 1.2),
    # (copy_x_test['ROI'] >= 1.2) & (copy_x_test['ROI'] < 1.8),
    # (copy_x_test['ROI'] >= 1.8) & (copy_x_test['ROI'] < 2.4),
    # (copy_x_test['ROI'] >= 2.4)
    
    if(roi < 0.6):
        st.write(f'The predicted Return on Investment is poor')
    elif(roi>=0.6 and roi < 1.2):
        st.write(f'The predicted Return on Investment is below average')
    elif(roi>=1.2 and roi <1.8):
        st.write(f'The predicted Return on Investment is average')
    elif(roi>=1.8 and roi <2.4):
        st.write(f'The predicted Return on Investment is good')
    else:
        st.write(f'The predicted Return on Investment is excellent')
    
        
