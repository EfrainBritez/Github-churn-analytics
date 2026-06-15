# Feature Importance and Retention Decision Report

## 1. Purpose

This report explains the feature selection methods used in the GitHub churn prediction project, in this specific case the data is scrapped from users from Paraguay, Argentina and Brazil, a total of 1500 users, compares their outputs, and connects the selected predictors to real retention decisions. The goal is to show conceptual understanding, not just code results.

## 2. Why feature selection matters

Feature selection is important for churn modeling because it helps identify the most useful signals and reduces noise. In a business context, stable and interpretable features let product and retention teams prioritize actions based on what the model actually uses.

## 3. Methods compared

Four selection methods were used:

- **Variance + correlation filter**: removes low-variance features and drops features that are highly correlated with others.
- **Decision tree importance**: selects features based on a single decision tree's split quality.
- **Random forest importance**: averages feature contributions across many trees.
- **Recursive feature elimination (RFE)**: removes the least important feature iteratively until the desired number remains.

Each method has strengths and weaknesses:

- Filters are simple and fast, but they ignore the target label beyond correlation structure.
- Tree-based importances capture interactions and non-linear split value, but can over-emphasize features that are directly tied to the label.
- RFE is more robust but computationally expensive and sensitive to the estimator used.

## 4. Selected features and agreement/disagreement

The dataset contains 1,500 labeled users. The different methods produced these feature sets:

- **Filtered features**: `account_age_days`, `repos_per_year`, `followers_following_ratio`, `active_repo_ratio`, `inactive_repo_ratio`, `avg_stars_per_repo`, `avg_forks_per_repo`, `repository_maintenance_ratio`
- **Decision tree selected**: `days_since_last_activity`, `account_age_days`, `repos_per_year`, `followers_following_ratio`, `active_repo_ratio`, `inactive_repo_ratio`
- **Random forest selected**: `followers_following_ratio`, `account_age_days`, `repos_per_year`, `avg_stars_per_repo`, `avg_forks_per_repo`, `active_repo_ratio`
- **RFE selected**: `days_since_last_activity`, `repos_per_year`, `followers_following_ratio`, `active_repo_ratio`, `inactive_repo_ratio`, `repo_activity_density`

### Agreement

Common predictors across methods are:

- `followers_following_ratio`
- `account_age_days`
- `repos_per_year`
- `active_repo_ratio`

These features are stable signals that consistently appear important for predicting churn. In business terms, they reflect account maturity, engagement level, and repository activity — all plausible churn drivers.

For build the model the data is splited into two, a set of 1200 to train and a set of 300 to verify the prediction accuracy

### Disagreement and why it happened

The main disagreement is the inclusion of `days_since_last_activity` and the exact presence of popularity metrics:

- The decision tree and RFE both include `days_since_last_activity`. That feature is effectively the label definition in this dataset, so its presence is leakage rather than genuine predictive insight.
- Random forest importance did not select `days_since_last_activity`, which is a positive sign because it avoids a leaky feature and focuses on behavior metrics instead.
- The filter method kept `inactive_repo_ratio` and `repository_maintenance_ratio`, but the random forest did not. This suggests those ratios are correlated with stronger predictors and may not add incremental value in the final model.

In short, the disagreement is largely due to label leakage and the different biases of each technique:

- **Decision tree** can overfit to a single strong predictor.
- **RFE** follows the chosen estimator and may retain features that are valuable in its own ranking process.
- **Filter-based selection** does not use the target directly and therefore may include correlated but less predictive features.

## 5. Real-world retention interpretation

A business-focused churn model should point to actions, not just scores. The most important features suggest the following retention interventions:

- `active_repo_ratio`: Users with low active repo ratios are inactive; they may benefit from re-engagement nudges such as personalized project suggestions or prompts to update a repository.
- `followers_following_ratio`: Users with low social visibility or influence may need help discovering communities, followers, or collaboration opportunities.
- `account_age_days`: Newer users may require onboarding support, while older users with falling activity may need reactivation campaigns.
- `repos_per_year`: Low repository creation rates indicate low commitment; retention teams can encourage smaller contributions or simpler tasks.
- `avg_stars_per_repo` and `avg_forks_per_repo`: These popularity metrics can help identify users who are more visible and thus more likely to stay if they receive success-based encouragement.

### Business decision link

The model should be used to flag users with a high churn probability and then apply tiered retention actions:

- low-cost intervention for borderline churn risks: email reminders, push notifications, or in-app prompts.
- moderate intervention for stronger churn signals: invitations to community events, tutorials, or help resources.
- high-touch intervention only if the predicted value justifies cost.

This is the capstone connection: feature importance tells us which user behaviors matter, and retention decisions are built around those behaviors.

## 6. Recommended model strategy

For a production-ready churn model, the safest feature set is one that:

1. excludes leaky features like `days_since_last_activity`,
2. includes stable predictors confirmed by multiple methods, and
3. can be explained to the retention team.

That means a good working feature set is:

- `account_age_days`
- `repos_per_year`
- `followers_following_ratio`
- `active_repo_ratio`
- optionally `avg_stars_per_repo` and `avg_forks_per_repo` if the model performance improves.

## 7. Conclusion

The feature selection analysis shows that the model is driven by engagement and repository activity. The strongest business insight is that retention efforts should focus on reactivating low-activity users and supporting users whose social or repository engagement is declining. These are concrete, actionable findings from the feature importance work.