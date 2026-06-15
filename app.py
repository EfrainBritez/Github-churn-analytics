from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from features import compute_features_from_user
from model_rf import load_model, predict

app = FastAPI()

# Load the model once at startup
model = load_model("models/model_rf.pkl")

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = None  # Set via environment variable if needed

# ----------------------------
# Request/Response models
# ----------------------------


class PredictionResult(BaseModel):
    username: str
    churn_probability: float
    churned: bool
    churn_percentage: str


# ----------------------------
# Endpoints
# ----------------------------


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the prediction form page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GitHub Churn Predictor</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
                max-width: 500px;
                width: 90%;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 28px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
            }
            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
                transition: border-color 0.2s;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 8px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result.success {
                background: #f0f7ff;
                border-left: 4px solid #667eea;
            }
            .result.error {
                background: #fff3f3;
                border-left: 4px solid #e74c3c;
            }
            .result-title {
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 16px;
            }
            .result.success .result-title {
                color: #667eea;
            }
            .result.error .result-title {
                color: #e74c3c;
            }
            .result-content {
                font-size: 14px;
                line-height: 1.6;
                color: #555;
            }
            .churn-percentage {
                font-size: 24px;
                font-weight: bold;
                margin: 15px 0;
            }
            .churn-low { color: #27ae60; }
            .churn-medium { color: #f39c12; }
            .churn-high { color: #e74c3c; }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                font-weight: 600;
            }
            .loading.show {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 GitHub Churn Predictor</h1>
            <div class="subtitle">Enter a GitHub username to predict churn risk</div>
            
            <form onsubmit="predictChurn(event)">
                <div class="form-group">
                    <label for="username">GitHub Username:</label>
                    <input type="text" id="username" name="username" placeholder="e.g., torvalds" required>
                </div>
                <button type="submit">Predict Churn</button>
            </form>
            
            <div class="loading" id="loading">⏳ Loading...</div>
            <div class="result" id="result"></div>
        </div>

        <script>
            async function predictChurn(event) {
                event.preventDefault();
                const username = document.getElementById('username').value.trim();
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                
                loadingDiv.classList.add('show');
                resultDiv.classList.remove('show');
                resultDiv.innerHTML = '';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username })
                    });
                    
                    loadingDiv.classList.remove('show');
                    
                    if (!response.ok) {
                        const error = await response.json();
                        resultDiv.className = 'result error show';
                        resultDiv.innerHTML = `
                            <div class="result-title">⚠️ Error</div>
                            <div class="result-content">${error.detail || 'Failed to predict churn'}</div>
                        `;
                        return;
                    }
                    
                    const data = await response.json();
                    
                    let riskColor = 'churn-low';
                    if (data.churn_probability > 0.6) {
                        riskColor = 'churn-high';
                    } else if (data.churn_probability > 0.3) {
                        riskColor = 'churn-medium';
                    }
                    
                    resultDiv.className = 'result success show';
                    resultDiv.innerHTML = `
                        <div class="result-title">✅ Prediction for ${data.username}</div>
                        <div class="result-content">
                            <div class="churn-percentage ${riskColor}">${data.churn_percentage}</div>
                            <div><strong>Status:</strong> ${data.churned ? '🚨 High Risk' : '✨ Low Risk'}</div>
                        </div>
                    `;
                } catch (error) {
                    loadingDiv.classList.remove('show');
                    resultDiv.className = 'result error show';
                    resultDiv.innerHTML = `
                        <div class="result-title">⚠️ Error</div>
                        <div class="result-content">${error.message}</div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResult)
def predict_churn_by_username(request: dict[str, str]) -> Any:
    """Fetch GitHub user data and predict churn probability."""
    username = request.get("username", "").strip()
    
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    try:
        # Fetch user data from GitHub API
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        user_response = requests.get(
            f"{GITHUB_API_BASE}/users/{username}",
            headers=headers,
            timeout=10
        )
        
        if user_response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found on GitHub")
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"GitHub API error: {user_response.status_code}")
        
        user_data = user_response.json()
        
        # Fetch repositories for the user
        repos_url = f"{GITHUB_API_BASE}/users/{username}/repos"
        repos_response = requests.get(repos_url, headers=headers, params={"per_page": 100}, timeout=10)
        
        if repos_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch repositories: {repos_response.status_code}")
        
        repos_data = repos_response.json()
        
        # Compute features from the raw user data
        features = compute_features_from_user(user_data, repos_data)
        
        # Predict using the model
        prediction = predict(model, features)
        
        churn_prob = prediction["churn_probability"]
        is_churned = prediction["churned"]
        
        # Format the percentage
        churn_percentage = f"{churn_prob * 100:.1f}%"
        
        return PredictionResult(
            username=username,
            churn_probability=churn_prob,
            churned=is_churned,
            churn_percentage=churn_percentage
        )
    
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch GitHub data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
