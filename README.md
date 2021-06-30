# Udacity_Data_Science
**Airbnb was born in 2007 when two Hosts welcomed three guests to their San Francisco home. It now has 4 million hosts providing unique places and staying experiences for users. Airbnb travelers can sleep in a cozy treehouse in the rainforest, or have a brunch in a British Castle for a Downton Abbey-style getaway.

According to the official site, users on Airbnb can book a place to stay in 100,000+ cities across 220+ countries. I use the Kaggle New User Booking Project to solve the following problems:  
1. Which features play important roles when a new user’s book his/her first travel destination?  
2. What are the features distributions?  
3. Which country will a new user make as the first booking?  

By accurately predicting the first travel destination, Airbnb can share more personalized content to each new user, increase user experience, decrease the average time to first booking, and better forecast demand.

Data source: [Kaggle Airbnb New User Booking Data](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.  

The training and test sets are split by dates. In the test set,I will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.   
 
File descriptions  
train_users.csv - the training set of users  
test_users.csv - the test set of users  
id: user id  
date_account_created: the date of account creation  
timestamp_first_active: timestamp of the first activity, note that it can be earlier than date_account_created or   date_first_booking because a user can search before signing up  
date_first_booking: date of first booking  
gender  
age  
signup_method  
signup_flow: the page a user came to signup up from  
language: international language preference  
affiliate_channel: what kind of paid marketing  
affiliate_provider: where the marketing is e.g. google, craigslist, other  
first_affiliate_tracked: whats the first marketing the user interacted with before the signing up  
signup_app  
first_device_type  
first_browser  
country_destination: this is the target variable you are to predict  

sessions.csv - web sessions log for users  
user_id: to be joined with the column 'id' in users table  
action  
action_type  
action_detail  
device_type  
secs_elapsed   

countries.csv - summary statistics of destination countries in this dataset and their locations  
age_gender_bkts.csv - summary statistics of users' age group, gender, country of destination  
sample_submission.csv - correct format for submitting your predictions  **
