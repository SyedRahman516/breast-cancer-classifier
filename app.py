import flask
import os
import pickle
import pandas as pd
from skimage import io
from skimage import transform
# Import our libraries

# Pandas and numpy for data wrangling
import pandas as pd
import numpy as np

# Seaborn / matplotlib for visualization
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
# %matplotlib inline

# Import the trees from sklearn
from sklearn import tree

# Helper function to split our data
from sklearn.model_selection import train_test_split

# Helper fuctions to evaluate our model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import make_scorer
# Helper function for hyper-parameter turning.
from sklearn.model_selection import GridSearchCV

# Import our Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Import our Random Forest
from sklearn.ensemble import RandomForestClassifier
#Import out KNeighbors
from sklearn.neighbors import KNeighborsClassifier
import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

app = flask.Flask(__name__, template_folder='templates')

path_to_model = 'models/model.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'
path_to_image_classifier = 'models/image-classifier.pkl'

with open(path_to_model, 'rb') as f:
    model = pickle.load(f)

@app.route('/placeholder', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])

        # Make a prediction
        predictions = model.predict(X)

        # Get the first and only value of the prediction.
        prediction = predictions[0]

        # Get the predicted probabs
        predicted_probas = model.predict_proba(X)

        # Get the value of the first, and only, predicted proba.
        predicted_proba = predicted_probas[0]

        # The first element in the predicted probabs is % democrat
        precent_democrat = predicted_proba[0]

        # The second elemnt in predicted probas is % republican
        precent_republican = predicted_proba[1]


        return flask.render_template('index.html',
            input_text=user_input_text,
            result=prediction,
            precent_democrat=precent_democrat,
            precent_republican=precent_republican)




@app.route('/input_values/', methods=['GET', 'POST'])
def input_values():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('input_values.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        var_one = flask.request.form['input_variable_one']
        var_two = flask.request.form['another-input-variable']
        var_three = flask.request.form['third-input-variable']

        list_of_inputs = [var_one, var_two, var_three]

        return(flask.render_template('input_values.html',
            returned_var_one=var_one,
            returned_var_two=var_two,
            returned_var_three=var_three,
            returned_list=list_of_inputs))

    return(flask.render_template('input_values.html'))


@app.route('/images/')
def images():
    return flask.render_template('images.html')


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('classify_image.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            # Read the image using skimage
            img = io.imread(file)

            # Resize the image to match the input the model will accept
            img = transform.resize(img, (28, 28))

            # Flatten the pixels from 28x28 to 784x0
            img = img.flatten()

            # Get prediction of image from classifier
            predictions = image_classifier.predict([img])

            # Get the value of the prediction
            prediction = predictions[0]

            return flask.render_template('classify_image.html', prediction=str(prediction))

    return(flask.render_template('classify_image.html'))

@app.route('/', methods=["GET", "POST"])
def test():
    features = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]

    true_diagnosis = None

    if flask.request.method == "GET":
        df = pd.read_csv('./static/data.csv')
        df = df.drop(['Unnamed: 32','id'],axis = 1)

        # THIS IS OUR OUTPUT
        target = ['diagnosis']
        X = df[features]
        y = df['diagnosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
        random_row = X_test.sample(n=1)

        true_diagnosis = y_test.loc[random_row.index.values[0]]

        if true_diagnosis == 'B':
            true_diagnosis = 'benign'
        else:
            true_diagnosis = 'malignant'
        print(true_diagnosis)

        random_row = random_row.values[0]


    if flask.request.method == "POST":
        random_row = []
        for feature in features:
            item = flask.request.form[feature]
            random_row.append(item)

    data = zip(features, random_row)
    pred = model.predict(np.array([random_row]))


    # print(random_row)
    if pred[0] == 'B':
        pred = 'benign'
    else:
        pred = 'malignant'

    return flask.render_template('test.html', data=data, pred=pred,true_diagnosis=true_diagnosis)


if __name__ == '__main__':
    app.run(debug=True)
