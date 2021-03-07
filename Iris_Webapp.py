import streamlit as st 
import numpy as np
import pickle

knnmodel=pickle.load(open('knnmodel.pkl','rb'))

def classify(output):
    if output == 'Iris-setosa':
        st.image('setosa.jpg', caption='Iris-setosa')
    elif output == 'Iris-versicolor':
        st.image('Versicolor.jpg',caption='Iris-versicolor')
    else:
        st.image('virginica.jpg',caption='Iris-virginica')

def main():

    body_temp = """
    <style> body{background-color: #202020} </style>
     """

    st.markdown(body_temp,unsafe_allow_html=True)

    html_temp = """
    <div style="background-color:#4D774E ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Flower Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Sepallen = st.text_input('Sepal Length', value="Type SepalLength")
    Sepalwid = st.text_input('Sepal Width', value="Type Sepalwidth")
    Petallen = st.text_input('Petal Length', value="Type PetalLength")
    Petalwid = st.text_input('Petal Width', value="Type Petalwidth")

    user_inputs = np.array([[Sepallen, Sepalwid, Petallen, Petalwid]])

    if st.button("Classify"):
        output = knnmodel.predict(user_inputs)
        st.success('This is '+ output[0])
        classify(output[0])
    
if __name__ == '__main__':
    main()