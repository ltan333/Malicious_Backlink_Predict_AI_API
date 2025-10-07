# 🚀 Welcome To THD

# 🚨 Malicious Backlink Prediction API

This project provides an **enhanced API** for detecting **malicious backlinks** using a pre-trained AI model (PhoBERT).  
The application has been **refactored** into a clean, modular structure for better maintainability and scalability.  
Optimized for **Linux (Ubuntu)** with **GPU acceleration**.

---

## 📁 Project Structure

### File name project: "Malicious_Backlink_Predict_AI_API_Refactor"
```
Malicious_Backlink_Predict_AI_API_Refactor/
├── app/                                    # Main application directory
│   ├── main.py                            # Application entry point
│   ├── api/                               # API layer
│   │   └── api_server.py                  # FastAPI application setup
│   ├── cores/                             # Core functionality
│   │   ├── config.py                      # Configuration settings
│   │   ├── logging.py                     # Logging configuration
│   │   └── security.py                    # JWT authentication
│   ├── models/                            # AI models and loading
│   │   ├── load_models.py                 # PhoBERT model loader
│   │   ├── load_vncorenlp.py              # Vietnamese NLP processor
│   │   └── models_phobert/                # Model files
│   │       ├── phobert_base_v9/            # PhoBERT model weights
│   │       └── vncorenlp/                  # Vietnamese NLP models
│   ├── routers/                           # API route handlers
│   │   ├── routes_auth.py                 # Authentication endpoints
│   │   └── routes_predict.py              # Prediction endpoints
│   ├── schemas/                           # Pydantic data models
│   │   └── schemas.py                     # Request/Response schemas
│   ├── services/                          # Business logic services
│   │   ├── cache_service.py               # Homepage caching
│   │   ├── redirect_service.py            # Redirect detection
│   │   └── scraping_service.py            # Web scraping & content analysis
│   ├── utils/                             # Utility functions
│   │   └── preprocessing.py               # Text preprocessing
│   ├── datasets/                          # Custom dataset classes
│   │   └── custom_dataset.py              # Text dataset for inference
│   ├── homepage_cache/                    # Cached homepage classifications
│   │   └── homepage_cache.json
│   ├── logs/                              # Application logs
│   │   └── api_server.log
│   └── tests/                             # Test files
│       └── test_cuda.py
├── docker-compose.yml                     # Docker Compose configuration
├── Dockerfile                             # Docker image definition
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🆕 What's New in the Refactored Version

### ✨ **Enhanced Features**
- **🔧 Modular Architecture**: Clean separation of concerns with organized modules
- **🚀 Enhanced Scraping**: Improved web scraping with Playwright and HTTP/2 support
- **🛡️ Advanced Security**: JWT-based authentication with token validation
- **📊 Smart Caching**: Intelligent homepage caching for faster predictions
- **🔄 Redirect Detection**: Advanced redirect parameter detection and handling
- **🌐 Multi-format Support**: Enhanced PDF and HTML content processing
- **⚡ Performance Optimized**: Connection pooling and concurrent processing
- **📝 Comprehensive Logging**: Detailed logging for debugging and monitoring

### 🏗️ **Architecture Improvements**
- **Separation of Concerns**: Each module has a specific responsibility
- **Dependency Injection**: Clean dependency management
- **Error Handling**: Robust error handling throughout the application
- **Type Safety**: Full type hints for better code quality
- **Async/Await**: Fully asynchronous for better performance

---

## 🚀 Getting Started

### 🔧 Requirements
- **Python** `3.10.10`
- **PyTorch** `2.2.2 + CUDA 12.1`
- **Linux (Ubuntu)** with **NVIDIA GPU** (recommended)
- **Java 21** (for VnCoreNLP)

---

### 🛠️ Installation

1. **Clone the repository or Download project**
   ```bash
   git clone https://github.com/ltan333/Malicious_Backlink_Predict_AI_API.git
   cd Malicious_Backlink_Predict_AI_API_Refactor
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install PyTorch with CUDA**
   ```bash
   pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Test CUDA availability**
   ```bash
   python3 app/tests/test_cuda.py
   ```

6. **Run the API Server**
   ```bash
   python3 app/main.py
   ```

   Or using the module approach:
   ```bash
   python3 -m app.main
   ```

The API will be available at:  
📍 `http://localhost:8000`

---

## 📚 API Documentation

### 🔐 Authentication

First, obtain an access token:

```bash
curl -X POST "http://localhost:8000/get-access-token" \
     -H "Content-Type: application/json" \
     -d '{"api_key": "jlG7BdO4V8vZF2yWO02XWzETK36Rbu5W45h5acrARZV6Kz75148r90D9xRYwkex9"}'
```

### 🎯 Prediction Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -d '[
       {
         "domain": "example.com",
         "backlink": "https://example.com/page",
         "title": "Sample Title",
         "description": "Sample description"
       }
     ]'
```

### 📋 Response Format

```json
[
  {
    "domain": "example.com",
    "backlink": "https://example.com/page",
    "label": "An toàn",
    "score": 0.95
  }
]
```

### 🏷️ Label Types

- **"An toàn"**: Safe/legitimate content
- **"Cờ bạc"**: Gambling content
- **"Phim lậu"**: Pirated movie content
- **"Quảng cáo bán hàng"**: Commercial advertising

---

## 🐳 Getting Started with Docker Compose

This section guides you through:

✅ Installing Docker & Docker Compose  
✅ Building & Running your app  
✅ Stopping the service  

---

## 🐧 For **Ubuntu (Linux)**

### ✅ Step 1: Install Docker Engine

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```

Add Docker's official GPG key:

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

Set up the Docker repository:

```bash
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg]   https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"   | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Install Docker Engine:

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Docker installed correctly:

```bash
docker --version
```

> You should see something like: `Docker version 24.x.x, build xxxxx`

---

### ✅ Step 2: Install Docker Compose (CLI wrapper, optional)

Docker Compose v2 is now included in Docker as `docker compose` (with a **space**).

If you still want the legacy `docker-compose` (with a hyphen):

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

---

### ✅ Step 3: Run Your App

Make sure your project has:

* `Dockerfile`
* `docker-compose.yml`

In your project root directory:

```bash
cd Malicious_Backlink_Predict_AI_API_Refactor
docker-compose up --build
```

Or
```bash
cd Malicious_Backlink_Predict_AI_API_Refactor
docker-compose up -d --build # Runs the containers in the background (detached mode)
```

The API will be available at:  
📍 `http://localhost:8000`

---

### 🛑 To Stop the Service

```bash
docker-compose down
```

---

## 🪟 For **Windows**

### ✅ Step 1: Install Docker Desktop

1. Download from:  
   👉 [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2. Run the installer → Follow setup

3. Enable:

   * WSL 2 backend (recommended for performance)
   * Linux containers (default)

4. Open Docker Desktop and wait for it to start.

5. Open **Command Prompt**, **PowerShell**, or **WSL** terminal and verify:

```bash
docker --version
docker-compose --version
```

---

### ✅ Step 2: Build and Run the App

Navigate to your project folder using terminal:

```bash
cd Malicious_Backlink_Predict_AI_API_Refactor
docker-compose up --build
```

Or
```bash
cd Malicious_Backlink_Predict_AI_API_Refactor
docker-compose up -d --build # Runs the containers in the background (detached mode)
```

The API will be available at:  
📍 `http://localhost:8000`

---

### 🛑 To Stop the Service

```bash
docker-compose down
```

---

## 🔧 Configuration

### Environment Variables

The application can be configured through `app/cores/config.py`:

```python
# Server Configuration
SERVER_VERSION = "v1.0"
SERVER_PORT = 8000
SERVER_HOST = "0.0.0.0"

# Model Configuration
MAX_BATCH_SIZE = 64

# Authentication
API_SECRET_KEY = "your-secret-key"
ACCESS_TOKEN_EXPIRE_SECOND = 86400  # 24 hours

# Spam Detection
SPAM_LABELS = ("Cờ bạc", "Phim lậu", "Quảng cáo bán hàng")
```

---

## 📦 Notes

- Download Model weights and VnCoreNLP as instructed in `app/models/models_phobert/download_models.txt`
- Download Configs as instructed in `app/cores/download_configs.txt`
- The API is designed to handle batch predictions and supports intelligent caching via `homepage_cache`
- Enhanced with advanced web scraping capabilities using Playwright
- Supports both static and dynamic content analysis
- Includes comprehensive redirect detection and handling

---

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from app.api.api_server import app
print('✅ API server imports successfully')
"
```

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements or bug fixes!

---

## 📜 License

MIT License © 2025