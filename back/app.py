# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from joblib import load
from flask_cors import CORS

# Create Flask application
app = Flask(__name__)

# Load the trained Random Forest model
# Assuming you've already trained the model and saved it as 'random_forest_model.pkl'
model = load('random_forest_model.joblib')

# Define a function for preprocessing
def preprocess_data(df):
    # Your preprocessing steps here
    # ...
    unique_values_counts = {}
    for column in df.columns:
        unique_values = df[column].unique()
        value_counts = df[column].value_counts()
        unique_values_counts[column] = {'unique_values': unique_values, 'value_counts': value_counts}
        
    for column, values_counts_dict in unique_values_counts.items():
        print(f"\nColumn: {column}")
        print("Unique Values:", values_counts_dict['unique_values'])
        print("Value Counts:")
        print(values_counts_dict['value_counts'])

        
    #
    columns_to_remove = ['Username','User Profile Link','Term & Year','Specialization', 'Major', 'Department', 'User Profile Link','gmatA','gmatV','gmatQ','TOEFL Essay','Industry Exp', 'Intern Exp', 'Journal Pubs', 'ConfPubs']

    df = df.drop(columns=columns_to_remove)

    df.columns
    #
    df = df[pd.notnull(df['UG College'])]
    #
    mean_value = df['TOEFL Score'].mean()

    # Replace missing values with the mean
    df['TOEFL Score'].fillna(mean_value, inplace=True)
    df = df[df["CGPA"].notnull()]
    #
    mean_value = df['greV'].mean()

    # Replace missing values with the mean
    df['greV'].fillna(mean_value, inplace=True)

    mean_value = df['greQ'].mean()

    # Replace missing values with the mean
    df['greQ'].fillna(mean_value, inplace=True)

    mean_value = df['greA'].mean()

    # Replace missing values with the mean
    df['greA'].fillna(mean_value, inplace=True)

    le = preprocessing.LabelEncoder()
    df['UG label'] = le.fit_transform(df['UG College'].astype(str))

    df = df.drop(columns=[ 'UG College'])
    df['CGPA'] = 10*df['CGPA']/df['CGPA Scale']
    df['Topper CGPA'] = 10*df['Topper CGPA']/df['CGPA Scale']
    df = df.drop(columns=['CGPA Scale'])
    
    #
    def func0(program):
        if program.upper() == 'MS':
            return 0
        elif program.upper() == 'PHD':
            return 1
        else:
            return 2
    df['Program'] = df['Program'].apply(func0)
    #

    univ=['Carnegie Mellon University',
        'University of North Carolina Chapel Hill',
        'University of Illinois Urbana-Champaign',
        'University of California San Diego',
        'University of Minnesota Twin Cities',
        'Texas A and M University College Station',
        'Georgia Institute of Technology', 'University of Texas Austin',
        'University of Michigan Ann Arbor', 'Columbia University',
        'University of Maryland College Park', 'Arizona State University',
        'University of Cincinnati', 'Ohio State University Columbus',
        'North Carolina State University', 'Northeastern University',
        'University of Arizona', 'University of Wisconsin Madison',
        'SUNY Buffalo', 'Clemson University', 'University of Utah',
        'Rutgers University New Brunswick/Piscataway',
        'Virginia Polytechnic Institute and State University',
        'Stanford University', 'Massachusetts Institute of Technology',
        'California Institute of Technology',
        'University of Massachusetts Amherst',
        'University of California Irvine', 'Purdue University',
        'Cornell University', 'University of Florida',
        'University of Washington', 'Syracuse University',
        'University of Pennsylvania', 'University of Southern California',
        'University of Texas Dallas', 'University of Illinois Chicago',
        'George Mason University', 'Harvard University',
        'Johns Hopkins University', 'SUNY Stony Brook',
        'Northwestern University', 'New York University',
        'New Jersey Institute of Technology',
        'University of California Santa Barbara', 'Princeton University',
        'University of Colorado Boulder',
        'University of California Los Angeles',
        'University of North Carolina Charlotte',
        'University of Texas Arlington', 'University of California Davis',
        'Worcester Polytechnic Institute',
        'University of California Santa Cruz', 'Wayne State University']
    ranks = [48,90,75,45,156,189,72,65,21,18,136,215,561,101,285,344,262,56,340,701,353,262,327,2,1,5,305,219,111,14,167,68,581,15,129,501,231,801,3,24,359,31,39,751,135,13,206,35,90,301,104,601,367,484]
    print(len(univ), len(ranks))
    univdict = {univ[i]: ranks[i] for i in range(len(univ))} 
    print(univdict)
    #
    ranking = []
    # uniqueUnivs = list(df['University Name'].unique())
    # print((uniqueUnivs))
    for index, row in df.iterrows():
        # i = uniqueUnivs.index(row['University Name'])
        # print(row['University Name'])
        ranking.append(univdict[row['University Name']])
    print(len(ranking), len(ranks))
    df['ranking'] = ranking
    print('hiiiii')

    #
    df = df.drop(columns='University Name')
    df.reset_index(inplace=True)
    df = df.drop(columns=['index'])

    #
    # Assuming 'columns_to_normalize' is a list of columns you want to normalize
    columns_to_normalize = ['TOEFL Score', 'greV', 'greQ', 'greA', 'Topper CGPA', 'CGPA', 'ranking']

    # Select the columns and reshape for MinMaxScaler
    columns_data = df[columns_to_normalize].values.reshape(-1, len(columns_to_normalize))

    # Create a MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    normalized_columns = scaler.fit_transform(columns_data)

    # If needed, replace the original columns in your DataFrame with the normalized values
    df[columns_to_normalize] = normalized_columns

    #
    df = df[df["CGPA"].notnull()]

    print(df.columns)
    

    return df

CORS(app)


# Define an endpoint for handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the uploaded file as a DataFrame
        df = pd.read_json(file)
        username =df['Username']
        # Preprocess the data
        df = preprocess_data(df)

        # Make predictions using the trained model
        predictions = model.predict(df)

        # Add predictions to the DataFrame
        df['Admission_Prediction'] = predictions.tolist()
        df['Username']=username

        # Return the DataFrame as JSON
        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)