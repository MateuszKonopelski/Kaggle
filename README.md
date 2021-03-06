# Kaggle
*Kaggle is the most popular website hosting competitions for data scientists. This repository contains code and files which I developed for running and finished competitions mostly for learning purpose. While there are solutions, hints and hacks online (especially for older ones), I decided not to use them and work on my code independly. Only after I finished and upload all predictions, I use others solutions to learn new methods or understand where could I improve.*

## Titanic
  - Goal: Predict Survival
  - Methods used:
    - Gaussian Naive Bayes
    - Log Regression
    - Support Vector Machine
    - Random Forrest
    - K-Nearest Neighbours
    - Extreme Gradient Boosting
    - Combined methods
  - Best method: SVM(features=(Sex, SibSp, Parch, Embarked, isCabin, Title, AgeBuckets); C=0.8; kernel='rbf')
  - Score: 79.9%; top 16%
  
## Expedia
  - Goal: Predict Hotel Cluster
  - Methods used:
    - Most popular 5-clusters in for each srch_destination_id
  - Score: 30.005%, top 82%
  - Comment: This was a very specific competition where high result could be only achieved by leakage solution that couldn't work in real life + all best scores didn't use machine learning, therefore I didn't work more on it as it would be not beneficial for my data science leanring.
  
## AllState
  - Goal: Predict insurance choices
  - Comment: After initial data exploration I dropped this project because it resembles Titanic classification and therefore I wanted to learn machine leanring on some different problem. Although, I will go back to it, when I will be learning scikit-multilearn package. 
  
## Digit Recognizer
   - Goal: Recognize hand-written digits.
   - Top5 Method used (with cv score on train set):
     - SVC
     - KNeighborsClassifier
     - QuadraticDiscriminantAnalysis
     - GradientBoostingClassifier
     - RandomForestClassifier
     - Other methods tested: ExtraTreesClassifier, XGBClassifier, BaggingClassifier, LogisticRegression, GaussianNB, SGDClassifier, LinearDiscriminantAnalysis, DecisionTreeClassifier, RidgeClassifier, Perceptron, BernoulliNB, ExtraTreeClassifier, AdaBoostClassifier, Lars, ElasticNet, LassoLars, PassiveAggressiveRegressor
   - Score: 95.53% top 60%

## Random Acts of Pizza
- Goal: Predict if Reddit user recieved a Pizza (NLP problem)
- Methods used:
     - LogisticRegression
     - RidgeClassifier
     - LinearSVC
     - RandomForestClassifier
     - BaggingClassifier
     - KNeighborsClassifier
- Score: 57.46%; top 60%

## Bike Sharing Demand
- Goal: Forecast use of a city bikeshare system
- Coment: This is a regression task but also data is perfect to mastering visualization skills

