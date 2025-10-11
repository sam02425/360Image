#!/bin/bash

# Ubuntu Multi-View YOLOv8 Experiment Setup Script
# Optimized for RTX 3070 GPU systems
# Author: Multi-View Detection Research
# Date: 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons."
   exit 1
fi

log "Starting Ubuntu Multi-View YOLOv8 Experiment Setup"
log "Optimized for NVIDIA RTX 3070 GPU"

# System information
info "System Information:"
info "OS: $(lsb_release -d | cut -f2)"
info "Kernel: $(uname -r)"
info "Architecture: $(uname -m)"
info "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
info "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

# Check for NVIDIA GPU
log "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
else
    error "NVIDIA GPU or drivers not detected. Please install NVIDIA drivers first."
    error "Run: sudo ubuntu-drivers autoinstall"
    exit 1
fi

# Check CUDA installation
log "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    info "CUDA version: $CUDA_VERSION"
else
    warn "CUDA toolkit not found. Installing CUDA..."
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA toolkit
    sudo apt-get install -y cuda-toolkit-12-4 

    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    info "CUDA installed. Please restart your terminal or run: source ~/.bashrc"
fi

# Update system packages
log "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
log "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    htop \
    nvtop \
    tree \
    vim \
    tmux \
    screen

# Install additional libraries for computer vision
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-dev

# Create experiment directory
EXP_DIR="$HOME/multiview_experiment"
log "Creating experiment directory: $EXP_DIR"
mkdir -p "$EXP_DIR"
cd "$EXP_DIR"

# Create Python virtual environment
log "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
log "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
log "Installing Python dependencies..."
pip install \
    ultralytics>=8.0.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=1.3.0 \
    numpy>=1.21.0 \
    Pillow>=8.0.0 \
    tqdm>=4.60.0 \
    pyyaml>=6.0 \
    opencv-python>=4.5.0 \
    scikit-learn>=1.0.0 \
    jupyter>=1.0.0 \
    notebook>=6.0.0

# Create directory structure
log "Creating directory structure..."
mkdir -p datasets models results logs scripts

# Copy experiment script
log "Setting up experiment script..."
cat > scripts/ubuntu_multiview_experiment.py << 'EOF'
# The Python script content would be inserted here
# For now, we'll create a placeholder
print("Ubuntu Multi-View YOLOv8 Experiment Script")
print("Please copy the ubuntu_multiview_experiment.py script to this location")
EOF

# Create run script
log "Creating run script..."
cat > run_experiment.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run experiment
echo "Starting Multi-View YOLOv8 Experiment..."
echo "GPU Memory before start:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Run the experiment
python3 scripts/ubuntu_multiview_experiment.py "$@"

echo "Experiment completed!"
echo "GPU Memory after completion:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
EOF

chmod +x run_experiment.sh

# Create quick test script
log "Creating quick test script..."
cat > quick_test.sh << 'EOF'
#!/bin/bash

# Quick test with 10 epochs
echo "Running quick test (10 epochs)..."
./run_experiment.sh --quick-test
EOF

chmod +x quick_test.sh

# Create monitoring script
log "Creating monitoring script..."
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash

# Monitor GPU usage during training
echo "Monitoring GPU usage. Press Ctrl+C to stop."
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
EOF

chmod +x monitor_gpu.sh

# Create dataset download helper
log "Creating dataset download helper..."
cat > download_datasets.sh << 'EOF'
#!/bin/bash

# Dataset download helper
echo "Dataset Download Helper"
echo "====================="
echo ""
echo "Please place your dataset zip files in the datasets/ directory:"
echo "- dual_dataset.zip"
echo "- quad_dataset.zip"
echo "- octal_dataset.zip"
echo "- full_dataset.zip"
echo ""
echo "Or provide URLs when running the experiment:"
echo "./run_experiment.sh --dataset-urls URL1 URL2 URL3 URL4"
echo ""
echo "Current datasets directory:"
ls -la datasets/
EOF

chmod +x download_datasets.sh

# Create system check script
log "Creating system check script..."
cat > check_system.sh << 'EOF'
#!/bin/bash

echo "System Check for Multi-View YOLOv8 Experiment"
echo "============================================="
echo ""

# Check NVIDIA driver
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || echo "❌ NVIDIA driver not found"
echo ""

# Check CUDA
echo "CUDA Version:"
nvcc --version | grep "release" || echo "❌ CUDA not found"
echo ""

# Check Python and packages
echo "Python Environment:"
source venv/bin/activate
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
echo ""

# Check disk space
echo "Disk Space:"
df -h .
echo ""

# Check memory
echo "System Memory:"
free -h
echo ""

echo "✅ System check completed!"
EOF

chmod +x check_system.sh

# Create README
log "Creating README..."
cat > README.md << 'EOF'
# Ubuntu Multi-View YOLOv8 Experiment

This directory contains everything needed to run multi-view YOLOv8 experiments on Ubuntu with RTX 3070.

## Quick Start

1. **Check System**: `./check_system.sh`
2. **Download Datasets**: Place zip files in `datasets/` or use `./download_datasets.sh`
3. **Run Experiment**: `./run_experiment.sh`
4. **Quick Test**: `./quick_test.sh` (10 epochs only)

## Directory Structure

```
multiview_experiment/
├── venv/                 # Python virtual environment
├── datasets/             # Dataset zip files and extracted data
├── models/               # Trained model weights
├── results/              # Training results and plots
├── logs/                 # Experiment logs
├── scripts/              # Python scripts
├── run_experiment.sh     # Main experiment runner
├── quick_test.sh         # Quick test (10 epochs)
├── monitor_gpu.sh        # GPU monitoring
├── check_system.sh       # System verification
└── download_datasets.sh  # Dataset helper
```

## GPU Optimization (RTX 3070)

- **Batch Size**: 32 (optimized for 8GB VRAM)
- **Workers**: 8 (multi-core CPU utilization)
- **Mixed Precision**: Enabled
- **Memory Management**: Optimized for CUDA

## Monitoring

- Monitor GPU usage: `./monitor_gpu.sh`
- Check system status: `./check_system.sh`
- View logs: `tail -f logs/ubuntu_experiment.log`

## Troubleshooting

1. **CUDA Issues**: Ensure NVIDIA drivers and CUDA toolkit are installed
2. **Memory Issues**: Reduce batch size in the script
3. **Permission Issues**: Ensure scripts are executable (`chmod +x *.sh`)

## Results

After completion, check:
- `results/` for training plots and metrics
- `models/` for trained model weights
- `results/ubuntu_experiment_summary.json` for detailed results
EOF

# Final setup
log "Setting up environment activation..."
echo "" >> ~/.bashrc
echo "# Multi-View Experiment Environment" >> ~/.bashrc
echo "alias activate-multiview='cd $EXP_DIR && source venv/bin/activate'" >> ~/.bashrc

# Test PyTorch CUDA
log "Testing PyTorch CUDA installation..."
source venv/bin/activate
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

log "Setup completed successfully!"
info "Experiment directory: $EXP_DIR"
info "To activate environment: activate-multiview (after restarting terminal)"
info "Or manually: cd $EXP_DIR && source venv/bin/activate"
info "Run system check: ./check_system.sh"
info "Start experiment: ./run_experiment.sh"

echo ""
echo "🎉 Ubuntu Multi-View YOLOv8 Experiment Setup Complete!"
echo "📁 Directory: $EXP_DIR"
echo "🚀 Ready to run experiments on RTX 3070"
echo ""
echo "Next steps:"
echo "1. Place dataset zip files in datasets/ directory"
echo "2. Run: cd $EXP_DIR && ./check_system.sh"
echo "3. Run: ./run_experiment.sh"
echo ""