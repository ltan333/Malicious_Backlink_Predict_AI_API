# 🚀 Welcome To THD

# 🚨 Malicious Backlink Prediction API

This project provides an API for detecting **malicious backlinks** using a pre-trained AI model (PhoBERT).  
Optimized for **Linux (Ubuntu)** with **GPU acceleration**.

---

## 📁 Project Structure

### File name project: "Malicious_Backlink_Predict_AI_API"
```
.
├── Homepage_Cache/
│   └── homepage_cache.json     # Cache for homepage results
├── Logs/
│   └── api_server.log          # Cache for logs when expose API
├── Models/
│   └── phobert_base_v4         # Fine-tuned PhoBERT model weights (version 4)
├── api_server.py               # Main script to launch the API server (FastAPI)
├── const.py                    # Constants used throughout the project
├── test_cuda.py                # Script to test CUDA availability and GPU setup
├── requirements.txt            # Python dependency list
├── Dockerfile                  # Instructions to build the Docker image
├── docker-compose.yml          # Multi-service orchestration with Docker Compose
├── README.md                   # Project overview, setup instructions, and usage guide
├── .gitignore                  # Specifies files/folders to exclude from Git
└── Models_Configs.txt          # Instructions for downloading pre-trained model weights and configs
```
(venv) ubuntu@ai-model:~/Malicious_Backlink_Predict_AI_API$ tree
.
├── api_server.py
├── config
│   ├── const.py
│   └── download_configs.txt
├── docker-compose.yml
├── Dockerfile
├── homepage_cache
│   └── homepage_cache.json
├── models
│   ├── download_models.txt
│   ├── phobert_base_v8
│   │   ├── added_tokens.json
│   │   ├── bpe.codes
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.txt
│   └── vncorenlp
│       ├── models
│       │   ├── dep
│       │   │   └── vi-dep.xz
│       │   ├── ner
│       │   │   ├── vi-500brownclusters.xz
│       │   │   ├── vi-ner.xz
│       │   │   └── vi-pretrainedembeddings.xz
│       │   ├── postagger
│       │   │   └── vi-tagger
│       │   └── wordsegmenter
│       │       ├── vi-vocab
│       │       └── wordsegmenter.rdr
│       └── VnCoreNLP-1.2.jar
├── others
│   └── test_cuda.py
├── README.md
└── requirements.txt
---

# You can run this project with 2 ways on Linux(Ubuntu) or Windows OS:

# 🚀 Getting Started with Terminal

### 🔧 Requirements
- **Python** `3.10.12`
- **PyTorch** `2.2.2 + CUDA 12.1`
- **Linux (Ubuntu)** with **NVIDIA GPU**

---

### 🛠️ Installation

1. **Clone the repository or Download project**
   ```bash
   git clone https://github.com/ltan333/Malicious_Backlink_Predict_AI_API.git
   cd Malicious_Backlink_Predict_AI_API
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
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
   python3 test_cuda.py
   ```

6. **Expose API**
   ```bash
   python3 API_Server.py
   ```

# 🐳 Getting Started with Docker Compose

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

Add Docker’s official GPG key:

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
cd Malicious_Backlink_Predict_AI_API
docker-compose up --build
```

Or
```bash
cd Malicious_Backlink_Predict_AI_API
docker-compose up -d --build # Runs the containers in the background (detached mode)
```

The API will be available at:  
📍 `http://localhost:8000/predict`

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
cd Malicious_Backlink_Predict_AI_API
docker-compose up --build
```

Or
```bash
cd Malicious_Backlink_Predict_AI_API
docker-compose up -d --build # Runs the containers in the background (detached mode)
```

The API will be available at:  
📍 `http://localhost:8000/predict`

---

### 🛑 To Stop the Service

```bash
docker-compose down
```


## 📦 Notes

- Download Model weights and Configs as instructed in `Models_Configs.txt`.
- The API is designed to handle batch predictions and supports caching via `Homepage_Cache`.

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements or bug fixes!

---

## 📜 License

MIT License © 2025
