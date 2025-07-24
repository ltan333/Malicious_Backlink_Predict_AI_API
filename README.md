# 🚀 Welcome To THD

# 🚨 Malicious Backlink Prediction API

This project provides an API for detecting **malicious backlinks** using a pre-trained AI model (PhoBERT).  
Optimized for **Linux (Ubuntu)** with **GPU acceleration**.

---

## 🚀 Getting Started

### 🔧 Requirements
- **Python** `3.10.12`
- **PyTorch** `2.2.2 + CUDA 12.1`
- **Linux (Ubuntu)** with **NVIDIA GPU**

---

### 🛠️ Installation

1. **Clone the repository or Download project**
   ```bash
   git clone https://github.com/your-user/Malicious_Backlink_Predict_AI_API.git
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

4. **Test CUDA availability**
   ```bash
   python3 test_cuda.py
   ```

5. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Expose API**
   ```bash
   python3 API_Server.py
   ```

## 🐳 Run with Docker Compose


### 🧱 Build and Run
```bash
docker-compose up --build
```

The API will be available at:  
📍 `http://localhost:8000/predict`

### 🛑 Stop the service
```bash
docker-compose down
```

---

## 📁 Project Structure

```
.
├── API_Server.py               # Main API server script
├── app.py                      # Entry point (optional)
├── const.py                    # Constants used in the project
├── Models/                     # Pre-trained model files
├── Homepage_Cache/
│   └── homepage_cache.json     # Cache for homepage results
├── test_cuda.py                # Script to test CUDA support
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image setup
├── docker-compose.yml          # Docker services orchestration
├── README.md                   # Project documentation
└── Models_Configs.txt          # Instruction to download models
```

---

## 📦 Notes

- Download Model weights and Configs as instructed in `Models_Configs.txt`.
- The API is designed to handle batch predictions and supports caching via `Homepage_Cache`.

---

## 🤝 Contributing

Feel free to open issues or pull requests for improvements or bug fixes!

---

## 📜 License

MIT License © 2025
