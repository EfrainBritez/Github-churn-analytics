# GitHub Churn Analytics

A machine learning application that predicts whether a GitHub user is likely to become inactive (churn) based on their repository and activity features.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

## Features

- **Churn Prediction**: Uses a trained Random Forest model to predict user churn probability
- **Feature Engineering**: Automatically computes GitHub user features from API data
- **Web Interface**: Interactive HTML form for single user predictions
- **REST API**: JSON endpoints for programmatic access
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git
- Docker & Docker Compose (optional, for containerized deployment)
- GitHub Personal Access Token (optional, for higher API rate limits)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/EfrainBritiz/Github-churn-analytics.git
cd Github-churn-analytics
```

### 2. Create a Virtual Environment (Recommended)

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup

### Creating the .env File

The application uses environment variables to manage configuration. Follow these steps:

#### 1. Create a `.env` File

Create a new file named `.env` in the project root directory (same level as `app.py`).

#### 2. Add Environment Variables

Add the following content to your `.env` file:

```
GITHUB_TOKEN=your_personal_access_token_here
PORT=8000
```

### Getting a GitHub Personal Access Token

A GitHub Personal Access Token allows you to make authenticated requests to the GitHub API with higher rate limits.

#### Steps to Create a Token:

1. Go to [GitHub Settings - Personal Access Tokens](https://github.com/settings/tokens)
2. Click **"Generate new token"** (or "Generate new token (classic)")
3. Give your token a descriptive name (e.g., "Churn Analytics")
4. Select the following scopes:
   - `public_repo` - for reading public repository data
   - `user` - for reading user profile data
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't be able to see it again)

#### Adding the Token to .env

Replace `your_personal_access_token_here` in your `.env` file with the actual token:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### .env File Security

⚠️ **Important**: 
- The `.env` file is automatically ignored by Git and should **NEVER** be committed to the repository
- Keep your GitHub token private - treat it like a password
- If you accidentally commit the token, immediately revoke it from [GitHub Settings](https://github.com/settings/tokens)

## Running the Application

### Option 1: Run Locally with Uvicorn

#### 1. Ensure Virtual Environment is Activated

#### 2. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

#### 3. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000/
```

### Option 2: Run with Docker

#### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

The application will start on `http://localhost:8000`

#### 2. Stop the Container

```bash
docker-compose down
```

#### 3. Run in Detached Mode

```bash
docker-compose up -d --build
```

### Option 3: Build and Run Docker Image Manually

```bash
# Build the image
docker build -t github-churn-analytics .

# Run the container
docker run -p 8000:8000 github-churn-analytics
```

## Using the Application

### Web Interface

1. Open `http://localhost:8000/` in your browser
2. Enter a GitHub username in the form
3. Click **"Predict Churn"**
4. View the prediction result showing:
   - Churn probability (0-1)
   - Whether the user is predicted to churn (Yes/No)
   - Churn percentage

### Command Line / API Client

You can also make requests using `curl` or any HTTP client:

```bash
curl http://localhost:8000/predict_json?username=torvalds
```

## API Endpoints

### 1. Home / Web Interface

**GET** `/`

Returns an interactive HTML form for predictions.

**Response**: HTML page

### 2. Predict Churn (JSON)

**GET** `/predict_json`

**Query Parameters**:
- `username` (string, required): GitHub username to predict

**Response Example**:
```json
{
  "username": "torvalds",
  "churn_probability": 0.15,
  "churned": false,
  "churn_percentage": "15%"
}
```

### 3. Predict Churn (Form)

**POST** `/predict`

**Form Parameters**:
- `username` (string): GitHub username to predict

**Response**: Redirects to results page with prediction

### 4. Get User Features (JSON)

**GET** `/get_user_features`

**Query Parameters**:
- `username` (string, required): GitHub username

**Response Example**:
```json
{
  "username": "torvalds",
  "features": {
    "repos_created": 45,
    "avg_stars_per_repo": 250.5,
    ...
  }
}
```

### Error Responses

**400 Bad Request**: Missing or invalid username
```json
{
  "detail": "Username not provided"
}
```

**404 Not Found**: User not found on GitHub
```json
{
  "detail": "GitHub user not found"
}
```

**500 Internal Server Error**: Server error during prediction
```json
{
  "detail": "Error message here"
}
```

## Troubleshooting

### Issue: "Rate limit exceeded" Error

**Cause**: Making too many API requests to GitHub without authentication

**Solution**: 
1. Create a GitHub Personal Access Token (see [Environment Setup](#environment-setup))
2. Add it to your `.env` file as `GITHUB_TOKEN=your_token`
3. Restart the application

### Issue: "GITHUB_TOKEN not found" Warning

**Cause**: The `.env` file is missing or empty

**Solution**:
1. Create a `.env` file in the project root directory
2. Add your `GITHUB_TOKEN` (optional but recommended)
3. Restart the application

### Issue: "User not found" Error

**Cause**: The GitHub username doesn't exist or is misspelled

**Solution**:
1. Check the username spelling
2. Verify the user exists on GitHub
3. Try searching directly on github.com

### Issue: Port 8000 Already in Use

**Cause**: Another application is using port 8000

**Solution - Local**:
```bash
uvicorn app:app --port 8001
```

**Solution - Docker**:
```bash
docker-compose up -d -p "8001:8000"
```

### Issue: Model File Not Found

**Cause**: The trained model file `models/model_rf.pkl` is missing

**Solution**:
1. Ensure you have the `models/` directory in your project root
2. Train the model using the provided training scripts
3. Make sure the model file is named `model_rf.pkl`

### Issue: ModuleNotFoundError

**Cause**: Dependencies are not installed or virtual environment is not activated

**Solution**:
1. Activate your virtual environment (see [Installation](#installation))
2. Install dependencies: `pip install -r requirements.txt`
3. Verify with: `pip list`

## Project Structure

```
.
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── .env                     # Environment variables (create this)
├── models/
│   └── model_rf.pkl         # Trained Random Forest model
├── data/                    # Data directory
│   ├── raw/                 # Raw GitHub data
│   ├── processed/           # Processed data
│   └── filtered/            # Filtered features
├── reports/                 # Reports and documentation
└── tests/                   # Test files
```

## Additional Resources

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## License

This project is part of the IDS Final Project.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
