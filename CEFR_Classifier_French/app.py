import streamlit as st
from model import predict_french_difficulty, load_model_and_classifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import base64

logo_path = '/content/Logo_team_orange_DS&ML.png' 
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo_path, use_column_width=True)


# Load the model and classifier at the start of the application
model, tokenizer, classifier = load_model_and_classifier()

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Prediction", "Model Settings", "Other"])

with tab1:
    st.title("Prediction of the Difficulty of French Texts")

    st.markdown("""
    ## 🌟 Project Context
    Welcome to our application!
    
    We are Khadija Oukerroum and Léo Rongier, students at HEC Lausanne. In 2023, we embarked on an exciting journey for our Data Science and Machine Learning course. Our project's objective was to conceptualize, create, and train a machine learning model to predict the difficulty level of French texts. We invite you to join us in this adventure and experience our model's capabilities.
    
    ## 🔍 Methodology
    To train our model, we explored various Machine Learning techniques, including:
    - Logistic Regression with Finetuned-Embedding
    - Logistic Regression with Multilingual Embedding
    - Logistic Regression with TF-IDF
    - Logistic Regression with Bag-of-Words
    - KNN (K-Nearest Neighbors)
    - Decision Tree
    - Random Forest

    Each of these techniques was carefully tested to assess its performance in the prediction task.

    ## 📈 Results
    Our best accuracy score reached is 0.58, a significant milestone in the pursuit of efficient and precise language models.

    ## 📚 About this Application
    This Streamlit application is designed to allow users to interact with our model. You can:
    - Test the difficulty prediction on phrases or texts in French.
    - Explore the model settings and their impact on performance.
    - (If applicable) Download the training dataset we used to better understand our approach.

    ## 🚀 Start Exploring
    Use the tabs above to navigate the application and discover the available features.
    """)

with tab2:
    st.header("Predict Sentence Difficulty")
    user_input = st.text_area("Enter the sentence to evaluate:", key="prediction_input")
    if st.button("Predict", key="predict_button"):
        prediction = predict_french_difficulty(user_input, model, tokenizer, classifier)
        st.write(f"Predicted difficulty: {prediction}")

with tab3:
  # Load your data (adapt according to the correct path of your file)
  data = pd.read_csv('training_data.csv')
  train, test = train_test_split(data, test_size=0.2, random_state=42)

  # Function to run and evaluate a given model
  def evaluate_model(model, vectorizer, train_data, test_data, train_labels, test_labels):
      pipe = Pipeline([('vectorizer', vectorizer), ('classifier', model)])
      pipe.fit(train_data, train_labels)
      predictions = pipe.predict(test_data)
      precision = precision_score(test_labels, predictions, average='macro')
      recall = recall_score(test_labels, predictions, average='macro')
      f1 = f1_score(test_labels, predictions, average='macro')
      accuracy = accuracy_score(test_labels, predictions)
      return precision, recall, f1, accuracy

  # Streamlit widgets to allow the user to set the models' parameters
  st.title('Machine Learning Model Parameter Settings')

  # Parameters for Logistic Regression
  st.header('Logistic Regression')
  C_lr = st.slider('C (Regularization)', 0.01, 10.0, 1.0, key='C_lr')
  max_df_tfidf = st.slider('TF-IDF max_df', 0.0, 1.0, 0.5, key='max_df_tfidf')
  min_df_tfidf = st.slider('TF-IDF min_df', 0.0, 1.0, 0.1, key='min_df_tfidf')

  # Parameters for KNN
  st.header('K-Nearest Neighbors')
  n_neighbors = st.slider('Number of neighbors (n_neighbors)', 1, 20, 5, key='n_neighbors')

  # Parameters for Decision Tree
  st.header('Decision Tree')
  max_depth_dt = st.slider('Maximum depth (max_depth_dt)', 1, 32, 10, key='max_depth_dt')

  # Parameters for Random Forest
  st.header('Random Forest')
  n_estimators_rf = st.slider('Number of trees (n_estimators_rf)', 10, 300, 100, key='n_estimators_rf')
  max_depth_rf = st.slider('Maximum depth (max_depth_rf)', 1, 32, 10, key='max_depth_rf')

  # Button to run the models
  if st.button('Run models'):
      performance_df = pd.DataFrame(index=['precision', 'recall', 'f1-score', 'accuracy'])
      
      # Logistic Regression with TF-IDF
      tfidf = TfidfVectorizer(max_df=max_df_tfidf, min_df=min_df_tfidf)
      lr = LogisticRegression(C=C_lr)
      precision, recall, f1, accuracy = evaluate_model(lr, tfidf, train['sentence'], test['sentence'], train['difficulty'], test['difficulty'])
      performance_df['Logistic Regression'] = [precision, recall, f1, accuracy]
      
      # KNN
      bow = CountVectorizer()
      knn = KNeighborsClassifier(n_neighbors=n_neighbors)
      precision, recall, f1, accuracy = evaluate_model(knn, bow, train['sentence'], test['sentence'], train['difficulty'], test['difficulty'])
      performance_df['KNN'] = [precision, recall, f1, accuracy]
      
      # Decision Tree
      dt = DecisionTreeClassifier(max_depth=max_depth_dt)
      precision, recall, f1, accuracy = evaluate_model(dt, bow, train['sentence'], test['sentence'], train['difficulty'], test['difficulty'])
      performance_df['Decision Tree'] = [precision, recall, f1, accuracy]
      
      # Random Forest
      rf = RandomForestClassifier(n_estimators=n_estimators_rf, max_depth=max_depth_rf)
      precision, recall, f1, accuracy = evaluate_model(rf, bow, train['sentence'], test['sentence'], train['difficulty'], test['difficulty'])
      performance_df['Random Forest'] = [precision, recall, f1, accuracy]
      
      # Display results
      st.write(performance_df)

  st.write("Adjust the parameters and click 'Run models' to see the performances.")

with tab4: 
    # Load the DataFrame (make sure it's the correct path)
    data = pd.read_csv('training_data.csv')

    # Display the download button in the Streamlit application
    st.download_button(
        label="Download Training Data",
        data=data.to_csv(index=False),
        file_name='training_data.csv',
        mime='text/csv',
    )

    # Selection to sort by difficulty level
    difficulty_to_sort = st.selectbox('Sort by difficulty level:', options=sorted(data['difficulty'].unique()))

    # Filter the DataFrame based on the selected difficulty
    filtered_data = data[data['difficulty'] == difficulty_to_sort]

    # Display the filtered DataFrame
    # Streamlit will display a slider if the data exceeds the default visible height.
    st.dataframe(filtered_data)