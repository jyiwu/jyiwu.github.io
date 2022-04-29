import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 

#@st.cache(allow_output_mutation = True) 

st.markdown('# HELOC Risk Evaluation')
st.write('This is a simple app that help you to decide whether to accept or reject an credit card application.')
st.write('(Use slider to select Features and click on the Run Model button to see the result)')

# load model
def load_model():
	with open('LG_Model', 'rb') as model:
		loaded_model = pickle.load(model)
	return loaded_model

loaded_model = load_model()

# initialization
prediction = []
selections = []
features = []
coefs=[round(x,3) for x in loaded_model.coef_[0]]


# features
st.sidebar.markdown("## Feature Selection:")

x1=st.sidebar.slider('ExternalRiskEstimate', 0, 100, 0, 1)
x2=st.sidebar.slider('PercentTradesNeverDelq', 0, 100, 0, 1)
x3=st.sidebar.slider('AverageMInFile', 0, 400, 0, 1) 
x4=st.sidebar.slider('NumSatisfactoryTrades', 0, 100, 0, 1) 
x5=st.sidebar.slider('NumInqLast6M', 0, 100, 0, 1)
x6=st.sidebar.slider('NetFractionRevolvingBurden', 0, 300, 0, 1)
x7=st.sidebar.slider('PercentInstallTrades', 0, 100, 0, 1) 
x8=st.sidebar.slider('MaxDelq2PublicRecLast12M', 0, 50, 0, 1) 
x9=st.sidebar.slider('NumRevolvingTradesWBalance', 0, 100, 0, 1)
x10=st.sidebar.slider('NetFractionInstallBurden', 0, 600, 0, 1)
x11=st.sidebar.slider('MSinceMostRecentDelq', 0, 100, 0, 1)
selections = [int(x) for x in [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]]

features = ['ExternalRiskEstimate', 'PercentTradesNeverDelq', 'AverageMInFile',
       'NumSatisfactoryTrades', 'NumInqLast6M',
       'NetFractionRevolvingBurden', 'PercentInstallTrades',
       'MaxDelq2PublicRecLast12M', 'NumRevolvingTradesWBalance',
       'NetFractionInstallBurden', 'MSinceMostRecentDelq']

multi = []
for i in range(0,11):
        multi.append(selections[i]*coefs[i])

output_df = pd.DataFrame([selections, coefs, multi],
                         index=['input', 'coefficients', 'multiplication'],
                         columns=features).T
st.dataframe(output_df)

st.write(f"The intercept: {loaded_model.intercept_} \t The prediction: {sum(multi) + loaded_model.intercept_}")
#st.write(f"The prediction: {sum(multi) + loaded_model.intercept_}")


prediction = loaded_model.predict(pd.DataFrame([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]],
                                               columns=features))[0]





st.markdown('# Result Here:')

	# prediction result
def show_prediction():
        if st.button("Run Model"):
                pred= prediction
                if pred == 1:
                        st.markdown("#### This customer can be accepted!")
                else:
                        st.markdown("#### Reject!")
                st.write('**Accuracy:** 71.29%')
show_prediction()
















