# AI Training Supervisor

> An intelligent ML training supervision and recovery system designed to monitor machine learning workloads, detect training issues, evaluate experiments, and assist with controlled recovery.

---

## 📌 Overview

**AI Training Supervisor** is a modular system for managing and supervising machine learning training jobs.

Instead of simply starting a training script and waiting for it to finish, the system provides a supervision layer around the training process.

It is designed to:

- Manage training jobs
- Monitor training execution
- Track training state and progress
- Detect failures and abnormal conditions
- Perform controlled recovery actions
- Manage checkpoints
- Evaluate trained models
- Provide experiment information
- Provide API and dashboard interfaces
- Optionally use a local LLM through Ollama for diagnostic assistance

The core training workload runs on the local machine GPU, while the optional LLM advisor can run on a separate Raspberry Pi.

---

## 🧠 System Architecture

```text
                         AI TRAINING SUPERVISOR
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
        Training Engine      Supervisor Core      Monitoring
              │                   │                   │
              │                   ├── Job Queue      │
              │                   ├── Orchestrator   │
              │                   ├── Job Manager    │
              │                   └── Policies      │
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
             Recovery         Evaluation       Database
                 │                │                │
                 └────────────────┼────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
               Streamlit                    FastAPI
               Dashboard                      API
                    │
                    ▼
             Optional AI Advisor
                    │
                    │ LAN
                    ▼
             Raspberry Pi 5
                    │
                 Ollama
                    │
                    ▼
           Qwen2.5-Coder 1.5B
```

---

# ✨ Features

## Training Management

- PyTorch training support
- Scikit-learn training support
- TensorFlow training entry point
- GPU acceleration through CUDA
- Training configuration management
- Checkpoint support
- Training job lifecycle management

## Supervisor

The Supervisor layer manages the lifecycle of training jobs.

It is responsible for:

- Job submission
- Job queuing
- Priority management
- Job execution
- Job state tracking
- Training lifecycle coordination

### Priority Queue

Training jobs are managed using a thread-safe priority queue.

Higher-priority jobs are processed before lower-priority jobs.

For jobs with the same priority, job IDs are used to preserve ordering.

---

## 📊 Monitoring

The monitoring subsystem is designed to observe training execution.

Possible monitored information includes:

- Training progress
- Training state
- Errors
- Runtime information
- Resource information
- Checkpoint status
- Training logs

---

## 🔄 Recovery

The recovery subsystem provides controlled responses to training failures.

The design follows a safety-first approach:

```text
Training Failure
      │
      ▼
Detect Error
      │
      ▼
Analyze Failure
      │
      ▼
Check Recovery Policy
      │
      ├── Recoverable ──► Recovery Action
      │                       │
      │                       ▼
      │                  Resume Training
      │
      └── Unknown/Error ──► Stop Safely
```

Recovery actions should be validated by the Supervisor's policy layer before execution.

The LLM advisor does **not** directly execute arbitrary shell or PowerShell commands.

---

# 🤖 AI Advisor

The project can optionally connect to a local LLM through **Ollama**.

The LLM is intended to assist with:

- Error diagnosis
- Training failure analysis
- Recovery suggestions
- Configuration recommendations
- Experiment analysis

The recommended architecture is:

```text
Training Error
      │
      ▼
Supervisor
      │
      ▼
Advisor
      │
      ▼
Ollama API
      │
      ▼
Local LLM
      │
      ▼
Structured Recommendation
      │
      ▼
Policy Validation
      │
      ▼
Allowed Recovery Action
```

The LLM is treated as an **advisor**, not as an unrestricted system administrator.

---

# 🍓 Raspberry Pi LLM Server

The AI Advisor can optionally use a Raspberry Pi 5 as a separate LLM server.

Current tested configuration:

```text
Device: Raspberry Pi 5
RAM: 8 GB
Architecture: ARM64 / aarch64
LLM Server: Ollama
Model: Qwen2.5-Coder 1.5B
Port: 11434
```

Example architecture:

```text
Windows PC
RTX 3050
    │
    │ LAN / Wi-Fi
    ▼
Raspberry Pi 5
    │
    ▼
Ollama
    │
    ▼
Qwen2.5-Coder 1.5B
```

The Raspberry Pi component is optional.

The core training supervisor can operate without it.

> **Security:** Ollama should be exposed only to trusted LAN devices. Do not expose port `11434` directly to the public internet.

---

# 🖥️ Hardware Configuration

The primary development environment uses:

```text
CPU: Windows-compatible x86-64 system
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 6 GB
```

PyTorch CUDA verification:

```python
import torch

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Expected result:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| ML | Scikit-learn |
| Optional DL | TensorFlow |
| GPU | NVIDIA CUDA |
| API | FastAPI |
| Dashboard | Streamlit |
| Database | SQLite |
| ORM | SQLAlchemy |
| Configuration | YAML / `.env` |
| Testing | Pytest |
| LLM Server | Ollama |
| LLM | Qwen2.5-Coder |
| Containerization | Docker |
| Version Control | Git |

---

# 📁 Project Structure

```text
ai-training-supervisor/
│
├── advisor/
│   └── AI advisor and diagnostic logic
│
├── api/
│   └── FastAPI application and endpoints
│
├── artifacts/
│   └── Generated models, checkpoints and outputs
│
├── dashboard/
│   └── Streamlit dashboard
│
├── data/
│   └── Training and experiment data
│
├── database/
│   └── Database models and database logic
│
├── docs/
│   └── Project documentation
│
├── evaluation/
│   └── Model evaluation components
│
├── integrations/
│   └── External service integrations
│
├── logs/
│   └── Application and training logs
│
├── monitoring/
│   └── Training monitoring components
│
├── recovery/
│   └── Recovery and failure-handling logic
│
├── supervisor/
│   ├── job_queue.py
│   ├── orchestrator.py
│   └── Supervisor components
│
├── training/
│   ├── train_pytorch.py
│   ├── train_sklearn.py
│   └── examples/
│
├── tests/
│   └── Automated tests
│
├── utils/
│   └── Utility modules
│
├── app.py
├── config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai-training-supervisor
```

---

## 2. Create the Conda environment

```bash
conda create -n tfclean python=3.10
```

Activate it:

```bash
conda activate tfclean
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Configuration

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Configure the required values inside `.env`.

The project also uses:

```text
config.yaml
```

for application-level configuration.

Do not commit private credentials or secrets.

---

# 🧪 Verify the Installation

## Test PyTorch

Run:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

---

## Run Tests

```powershell
python -m pytest -q
```

---

## Test Core Modules

```powershell
python -c "import supervisor; import training; import monitoring; import recovery; import evaluation; import advisor; print('ALL CORE MODULES IMPORTED SUCCESSFULLY')"
```

---

# 🏋️ Run PyTorch Training

Run:

```powershell
python .\training\train_pytorch.py
```

Example:

```text
Device: cuda
Epoch 1/10
Epoch 2/10
Epoch 3/10
...
Epoch 10/10
```

If `Device: cuda` appears, PyTorch is using the NVIDIA GPU.

---

# ▶️ Run the Supervisor

Start the main application:

```powershell
python .\app.py
```

---

# 🌐 API

The project contains a FastAPI layer for programmatic interaction with the Supervisor.

Typical development startup:

```bash
uvicorn api.main:app --reload
```

The exact API entry point may depend on the current configuration.

API functionality can include:

- Training job management
- Job status
- Checkpoint information
- Recovery operations
- Supervisor status

---

# 📊 Dashboard

The project includes a Streamlit dashboard.

Start it with:

```bash
streamlit run dashboard/app.py
```

The dashboard can be used to visualize and interact with the training supervision system.

---

# 🧪 Testing Strategy

The project uses `pytest`.

Testing is divided into several levels:

```text
Unit Tests
    │
    ▼
Component Tests
    │
    ▼
Integration Tests
    │
    ▼
Training Tests
    │
    ▼
End-to-End Supervisor Tests
```

Important components to test include:

- Job Queue
- Orchestrator
- Training execution
- Monitoring
- Recovery
- Evaluation
- API
- AI Advisor
- Database operations

Run all tests:

```bash
python -m pytest -q
```

---

# 🔐 Safety Design

The Supervisor follows several safety principles.

### 1. No unrestricted LLM execution

The AI Advisor should return recommendations rather than arbitrary commands.

### 2. Policy validation

Recommendations are validated before recovery actions are executed.

### 3. Bounded recovery

Recovery attempts should be limited to a configured maximum.

### 4. Safe failure

Unknown or unsupported errors should cause the system to stop safely rather than repeatedly modifying the system.

### 5. Artifact preservation

Training checkpoints, configurations and generated artifacts should be preserved whenever possible.

---

# 🔬 Research Potential

The project can serve as a foundation for research into:

- Autonomous ML operations
- Automated training supervision
- ML failure detection
- Intelligent checkpoint recovery
- LLM-assisted ML debugging
- Resource-aware training
- Automated experiment management
- AI-assisted MLOps
- Fault-tolerant machine learning pipelines

Possible future research directions include:

```text
Failure Detection
       +
Training Monitoring
       +
LLM-based Diagnosis
       +
Policy-based Recovery
       +
Experiment Evaluation
       =
Intelligent ML Training Supervisor
```

---

# 🗺️ Future Development

Planned improvements include:

- [ ] Advanced GPU monitoring
- [ ] CPU/RAM monitoring
- [ ] Automatic checkpoint selection
- [ ] More recovery strategies
- [ ] Training anomaly detection
- [ ] Improved experiment tracking
- [ ] Model comparison dashboard
- [ ] Distributed training support
- [ ] Docker deployment
- [ ] Authentication for API
- [ ] Role-based access
- [ ] More LLM providers
- [ ] Improved Ollama integration
- [ ] Notification system
- [ ] Advanced experiment analytics

---

# 🐳 Docker

Docker support is planned for reproducible deployment.

The intended deployment architecture is:

```text
Docker
│
├── Supervisor
├── API
├── Dashboard
├── Database
└── Training Services
        │
        ▼
     GPU / CUDA
```

The Raspberry Pi Ollama server can remain a separate service.

---

# 📌 Current Development Status

The current development environment has been verified for:

- ✅ Python 3.10
- ✅ Conda environment
- ✅ PyTorch
- ✅ CUDA
- ✅ NVIDIA RTX 3050 6GB
- ✅ Core module imports
- ✅ PyTorch training
- ✅ Pytest
- ✅ Supervisor startup
- ✅ Priority job queue
- ✅ Raspberry Pi Ollama server
- ✅ Qwen2.5-Coder 1.5B

The project is under active development.

---

# 🤝 Contributing

Contributions are welcome.

General workflow:

```bash
git clone <YOUR_REPOSITORY_URL>

git checkout -b feature/your-feature

git add .

git commit -m "Add your feature"

git push origin feature/your-feature
```

Then open a Pull Request.

---

# 📜 License

This project is currently intended for educational, research and development purposes.

A formal open-source license can be added later.

---

# 👨‍💻 Author

**Suprakash Ghosh**

B.Tech — Computer Science & Engineering  
Artificial Intelligence & Machine Learning

---

## ⭐ Project

**AI Training Supervisor**

An intelligent supervision layer for machine learning training, monitoring, diagnosis and controlled recovery.
