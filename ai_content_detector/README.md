# AI Content Detection System

A hybrid AI-generated text and image detector for high-accuracy, low-cost, and production-scale deployment.

## 🎯 Project Overview

This system provides robust detection of AI-generated content with:
- **Text Detection**: GPT-3/4, Claude, Mistral output identification
- **Image Detection**: DALL·E, MidJourney, GAN-generated image detection
- **High Performance**: ≥90% F1 Score, ≤500ms latency
- **Production Ready**: FastAPI + Docker + Streamlit UI

## 🏗️ Architecture

```
[User Input] → [Preprocessing] → [Inference Engine] → [Postprocessing] → [Output]
                                      ├── Text Detector (RoBERTa + Perplexity)
                                      └── Image Detector (EfficientNet + Frequency CNN)
```

## 🚀 Quick Start

### Installation

```bash
git clone <repository>
cd ai_content_detector
pip install -r requirements.txt
```

### API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit UI

```bash
streamlit run ui/app.py
```

### Docker Deployment

```bash
docker build -t ai-content-detector .
docker run -p 8000:8000 ai-content-detector
```

## 📁 Project Structure

```
ai_content_detector/
├── src/
│   ├── text_detection/        # Text detection models
│   ├── image_detection/       # Image detection models
│   ├── preprocessing/         # Input preprocessing
│   ├── postprocessing/        # Output processing
│   └── utils/                 # Shared utilities
├── api/                       # FastAPI backend
├── ui/                        # Streamlit frontend
├── models/                    # Trained models
├── data/                      # Datasets
├── tests/                     # Unit tests
├── config/                    # Configuration files
└── scripts/                   # Training/evaluation scripts
```

## 🔤 Text Detection Features

- **Perplexity Analysis**: GPT-2 based scoring
- **Burstiness Detection**: Sentence variance analysis
- **Transformer Classification**: Fine-tuned RoBERTa
- **Multi-model Support**: GPT-3/4, Claude, Mistral detection

## 🖼️ Image Detection Features

- **CNN Classification**: EfficientNet-based detection
- **Frequency Analysis**: Spectral domain features
- **PRNU Detection**: Photo Response Non-Uniformity analysis
- **GAN Detection**: Supports various generative models

## 📊 Performance Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Text F1 Score | ≥90% | 🎯 |
| Image F1 Score | ≥90% | 🎯 |
| API Latency | ≤500ms | 🎯 |
| Model Size | ≤100MB | 🎯 |

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **ML Framework**: PyTorch, Transformers
- **API**: FastAPI
- **UI**: Streamlit
- **Deployment**: Docker
- **Models**: RoBERTa, EfficientNet, Custom CNNs

## 📈 Usage Examples

### Text Detection API

```python
import requests

response = requests.post("http://localhost:8000/detect/text", 
                        json={"text": "Your text here"})
result = response.json()
print(f"AI Generated: {result['is_ai']}, Confidence: {result['confidence']}")
```

### Image Detection API

```python
files = {"image": open("image.jpg", "rb")}
response = requests.post("http://localhost:8000/detect/image", files=files)
result = response.json()
print(f"AI Generated: {result['is_ai']}, Confidence: {result['confidence']}")
```

## 🧪 Evaluation

Run benchmarks:

```bash
python scripts/evaluate_models.py --dataset benchmark --mode both
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request