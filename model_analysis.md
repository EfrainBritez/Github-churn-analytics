# Model trained with random forest data
Classification Report:
              precision    recall  f1-score   support

           0     0.9330    0.7677    0.8423       254
           1     0.3516    0.6957    0.4672        46

    accuracy                         0.7567       300
   macro avg     0.6423    0.7317    0.6547       300
weighted avg     0.8439    0.7567    0.7848       300

# Key insights
1. The amount of people that might churn is expresed on the recall of type 1, the model predicts it correctly 69.57%, this result is good if we can prevent the churn of the users without spending too much money, for exmaple with a notification or an email.

2. The confusion matrix will show us that the correct predicted no churn has a great percentage, almost 93% is correct labeled as no-churn when it is.

3. The model is not accurate to predict if the user will churn, it only has a 35% of accuracy.

# Model trained with filtered data. filtered_features
Classification Report:
              precision    recall  f1-score   support

           0     0.9321    0.8110    0.8674       254
           1     0.3924    0.6739    0.4960        46

    accuracy                         0.7900       300
   macro avg     0.6623    0.7425    0.6817       300
weighted avg     0.8494    0.7900    0.8104       300

# Key insights
1. The recall decreases but the precision improves a little bit, from 35% to 39%

# Change in labels, as the label a days_since_last_activity is too "weight" I remove and re do the labeling and featurization.
After remove the feature in rf
Classification Report:
              precision    recall  f1-score   support

           0     0.9211    0.8268    0.8714       254
           1     0.3889    0.6087    0.4746        46

    accuracy                         0.7933       300
   macro avg     0.6550    0.7177    0.6730       300
weighted avg     0.8395    0.7933    0.8105       300

In filtered_features
Classification Report:
              precision    recall  f1-score   support

           0     0.9321    0.8110    0.8674       254
           1     0.3924    0.6739    0.4960        46

    accuracy                         0.7900       300
   macro avg     0.6623    0.7425    0.6817       300
weighted avg     0.8494    0.7900    0.8104       300

The accurracy of the model increases upto 79.3% and now also the ranking is of rf is more useful

feature,importance
followers_following_ratio,0.19994092439178587
account_age_days,0.15922313022283735
repos_per_year,0.1386438208268372
avg_stars_per_repo,0.11186778465434259
avg_forks_per_repo,0.10412880259576057
active_repo_ratio,0.08676465604865972
repo_activity_density,0.08319773845783748
repository_maintenance_ratio,0.059435898261917905
inactive_repo_ratio,0.056797244540021156

Also was remove on filtered.
In both the model automatically detects as leaky column "Active_repo_ratio"

# Results on api.py
RF:
{'churned': False, 'churn_probability': 0.2291}
Filtered:
{'churned': False, 'churn_probability': 0.1417}