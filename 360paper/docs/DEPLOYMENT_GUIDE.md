# Production Deployment Guide for EfficientFormer-YOLO Retail Detection

## System Requirements

### Hardware (Minimum)
- **GPU**: NVIDIA RTX 3080 or better (10GB+ VRAM)
- **CPU**: 8+ cores
- **RAM**: 32 GB minimum
- **Storage**: 100GB SSD

### Software
- Ubuntu 20.04+ or Windows 10/11
- CUDA 11.8+
- Python 3.8+
- Docker (optional but recommended)

## Installation

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv retail_detection_env
source retail_detection_env/bin/activate  # Linux/Mac
# or
retail_detection_env\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics timm albumentations opencv-python-headless
pip install numpy pandas matplotlib seaborn tqdm
```

### 2. Model Training
```bash
# Train with EfficientFormer-YOLO
python experiments/train_efficientformer_yolo.py \
    --yolo-model yolov8n \
    --batch-size 16 \
    --epochs 100 \
    --dataset data/datasets/quad_dataset \
    --output-dir models/production
```

### 3. Model Optimization for Deployment
```bash
# Convert to ONNX for faster inference
python -c "
from models.efficientformer_yolo_hybrid import export_to_onnx
export_to_onnx('models/production/best_model.pth', 'models/production/model.onnx')
"

# Quantize for edge deployment (optional)
python -c "
from models.efficientformer_yolo_hybrid import quantize_model
quantize_model('models/production/model.onnx', 'models/production/model_int8.onnx')
"
```

## Camera Setup for Retail Environment

### Optimal Camera Placement
1. **Height**: 2.5-3.5 meters above ground
2. **Angle**: 30-45° downward tilt
3. **Coverage**: 4 cameras per 100m² (quad configuration)
4. **Resolution**: Minimum 1080p, recommended 4K
5. **FPS**: 15-30 FPS for real-time detection

### Camera Calibration
```python
# Camera calibration script
python scripts/calibrate_cameras.py --camera-config config/cameras.yaml
```

## Performance Optimization

### 1. Batch Processing
```python
# Process multiple frames in batch
batch_size = 8  # Adjust based on GPU memory
```

### 2. Model Quantization
- INT8 quantization for 2-4x speedup
- FP16 mixed precision for 1.5x speedup
- TensorRT optimization for NVIDIA GPUs

### 3. Multi-GPU Deployment
```python
# Distribute across multiple GPUs
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

## Monitoring and Maintenance

### Performance Metrics
- **Inference Time**: Target < 50ms per frame
- **Accuracy**: Target > 95% mAP@0.5
- **False Positives**: Target < 2%
- **System Load**: GPU < 80%, CPU < 60%

### Logging
```python
# Enable comprehensive logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retail_detection.log'),
        logging.StreamHandler()
    ]
)
```

### Model Updates
1. Collect new product images monthly
2. Retrain quarterly with new data
3. A/B test new models before deployment
4. Keep backup of previous best model

## API Endpoints

### REST API
```python
# Flask API example
@app.route('/detect', methods=['POST'])
def detect_products():
    image = request.files['image']
    results = model.predict(image)
    return jsonify(results)
```

### WebSocket for Real-time
```python
# WebSocket for streaming
@socketio.on('frame')
def handle_frame(data):
    results = model.predict(data['frame'])
    emit('detection', results)
```

## Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Check camera focus and lighting
   - Verify model is using correct weights
   - Ensure input preprocessing matches training

2. **Slow Performance**
   - Enable mixed precision training
   - Reduce batch size
   - Use model quantization

3. **Memory Issues**
   - Reduce image resolution
   - Decrease batch size
   - Enable gradient checkpointing

## Security Considerations

1. **Data Privacy**
   - Implement edge processing when possible
   - Encrypt data transmission
   - Regular security audits

2. **Model Security**
   - Protect model weights
   - Implement access controls
   - Monitor for adversarial attacks

## Docker Deployment

### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for API
EXPOSE 5000

# Run the application
CMD ["python3", "api/app.py"]
```

### Docker Compose
```yaml
version: '3'
services:
  retail_detection:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Cloud Deployment Options

### AWS
- Use EC2 instances with GPU (g4dn or p3 series)
- Deploy with AWS Elastic Container Service (ECS)
- Use S3 for model storage and versioning

### Azure
- Use Azure Machine Learning service
- Deploy with Azure Kubernetes Service (AKS)
- Use Azure Blob Storage for model storage

### Google Cloud
- Use Google Kubernetes Engine (GKE)
- Deploy with Cloud Run for serverless
- Use Cloud Storage for model storage

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Kubernetes for orchestration
- Implement auto-scaling based on CPU/GPU utilization

### Vertical Scaling
- Use more powerful GPUs for higher throughput
- Increase RAM for larger batch sizes
- Use NVMe SSDs for faster data loading

## Support and Updates

For support and updates, please contact:
- Email: support@retail-detection.com
- Documentation: https://docs.retail-detection.com
- GitHub: https://github.com/your-org/retail-detection