from model_rf import load_model, predict

model = load_model("models/model_rf.pkl")

result = predict(
    model,
    {
        "days_since_last_activity": 30,
        "account_age_days": 2000,
        "repos_per_year": 5.0,
        "followers_following_ratio": 0.5,
        "active_repo_ratio": 0.8,
        "avg_stars_per_repo": 2.0,
        "avg_forks_per_repo": 0.4,
        "inactive_repo_ratio": 0.2,
        "repository_maintenance_ratio": 0.7
    }
)

print(result)