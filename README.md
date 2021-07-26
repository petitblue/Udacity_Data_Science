# Udacity_Data_Science Project 1
## What is Airbnb New Users' first Booking Destionation?

### Overview
Airbnb was born in 2007 when two Hosts welcomed three guests to their San Francisco home. It now has 4 million hosts providing unique places and staying experiences for users. Airbnb travelers can sleep in a cozy treehouse in the rainforest, or have a brunch in a British Castle for a Downton Abbey-style getaway.

### Project Motivation
According to the official site, users on Airbnb can book a place to stay in 100,000+ cities across 220+ countries. Its imposible to place all the contries information to the front page. The motivation of this project is to predict new users Top 5 first booking destinations. The recommendation schema will help personalize the content and increase user experience. It will also decrease the average time to first booking, and better forecast demand. 

### Specific problems I plan to solve
To predict users' first booking choice, I will need to explore the numerical data and categorical data.
Generally, I will see:
1. What are Airbnb users look like? (their age, gender and online behaviors)
2. How do new users grow in Airbnb?
3. How's their age, gender and other features influence their behavior
4. What are their Top 5 choices of destinations?
5. What's the prediction of results, and which features are more important in the prediction?

### Project Data source
Data source: [Kaggle Airbnb New User Booking Data](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)
As the data is over 100MB, I read the data from Kaggle. And the script is run in the kernel. It the data is required, please contact me for the data

### Project Outline
1. Data Exploration and Preprocessing
2. Data Visualization
3. Data Engineering
4. Modeling and Evaluation

### Library Used
Panda  
Numpy  
Sklearn  

### File description
**Airbnb new user booking prediction.ipynb**
This file contains all the python code of the project.  
**'Submission.csv'**  
Prediction result

### Summary of Analysis Results
1. The majority of users are between the age of 20 to 45, and the averaged age is 36.There are slightly more female users than male users. In general, Airbnb female users are younger than male in average.
2. Users grew through the year, and increased rapidly in 2013 and 2014. There are more new users in summer than winter. Users aged from 16-30 contributed the most new accounts created.
3. The majority of users signup through airbnb and facebook. Most users choose desktop to signup, but younger users prefer mobile and older users prefer tablet. As to browers, most users choose Chrome and safari but older users prefer IE.
4. The top 5 destinations are US, France, Britain, Italy, Spain.
5. The top 10 influential features are: date of account created and time of first activation, sec_elasped(the time spend in online action), age, gender, signup method, first affiliate tracked, first device type and first browser, signup flow.

### Prediction Result
This project uses GXBoost to build the prediction model. The NDCG score is 0.833  CV score is 0.823
The predicted results are exported a file 'Submission.csv'
The result earns a score of 0.86414 in Kaggle

### Blog Post Link 
Based on the EDA and modeling, I wrote an article: [What are Airbnb New User’s First Booking Destination ?](https://medium.com/@dengluyan/what-are-airbnb-new-users-first-booking-destination-9bf70ba50a57) to demonstrate my findings and solve the problems raised in the begining. 


### Reference
1. Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
2. Data Science in 5 Minutes: What is One Hot Encoding?https://www.educative.io/blog/one-hot-encodin
3. Kurtis Pykes Normalized discounted cumulative gain https://towardsdatascience.com/normalized-discounted-cumulative-gain-37e6f75090e9 
4. Kaggle — Airbnb New User Prediction(Top 24%) https://chriskang028.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%B0%88%E6%A1%88-kaggle-airbnb-new-user-prediction-top-24-6d337dbcb772

