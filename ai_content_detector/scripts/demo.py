#!/usr/bin/env python3
"""
Demo script for AI Content Detection System.
Shows basic usage of text and image detection capabilities.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_detection import TextEnsemble
from src.image_detection import ImageEnsemble
from src.utils import setup_logger
from PIL import Image
import time

# Setup logging
logger = setup_logger(__name__)


def demo_text_detection():
    """Demonstrate text detection capabilities."""
    print("\n" + "="*50)
    print("🔤 TEXT DETECTION DEMO")
    print("="*50)
    
    try:
        # Load text ensemble
        print("Loading text detection models...")
        text_ensemble = TextEnsemble()
        print("✓ Text models loaded successfully\n")
        
        # Example texts
        examples = {
            "AI-Generated (GPT-4 style)": """
            The intersection of artificial intelligence and human creativity represents 
            one of the most fascinating frontiers in modern technology. As machine 
            learning algorithms become increasingly sophisticated, they demonstrate 
            remarkable capabilities in generating content that closely mimics human 
            expression. This convergence raises profound questions about the nature 
            of creativity, authorship, and the evolving relationship between human 
            and artificial intelligence in creative endeavors.
            """.strip(),
            
            "Human-Written (Personal)": """
            I remember the first time I tried to explain machine learning to my grandmother. 
            She was 87 and had never used a computer, but she was curious about what I did 
            for work. I started with the usual technical explanations, but her eyes glazed 
            over immediately. Then I tried a different approach: "You know how you can tell 
            when someone is lying just by looking at their face? Well, I'm teaching computers 
            to notice patterns like that, but with numbers instead of faces." Her face lit up. 
            "Oh, like when I can tell if bread is going to be good just by the sound it makes 
            when I tap it?" Exactly, Grandma. Exactly.
            """.strip(),
            
            "AI-Generated (Technical)": """
            Neural networks utilize backpropagation algorithms to optimize weight parameters 
            through gradient descent methodologies. The computational complexity of deep 
            learning architectures requires efficient matrix operations and vectorized 
            implementations to achieve scalable performance. Convolutional layers extract 
            hierarchical feature representations while attention mechanisms enable selective 
            focus on relevant input components, facilitating improved generalization across 
            diverse datasets and task-specific applications.
            """.strip()
        }
        
        # Analyze each example
        for title, text in examples.items():
            print(f"📝 Analyzing: {title}")
            print(f"Text preview: {text[:100]}...")
            
            start_time = time.time()
            result = text_ensemble.predict(text)
            processing_time = (time.time() - start_time) * 1000
            
            prediction = "🤖 AI-Generated" if result['is_ai'] else "👤 Human-Written"
            confidence = result['confidence']
            
            print(f"Result: {prediction} (Confidence: {confidence:.1%})")
            print(f"Processing time: {processing_time:.1f}ms")
            
            # Show individual model results if available
            if 'individual_predictions' in result:
                print("Individual model results:")
                for model, pred in result['individual_predictions'].items():
                    model_pred = "AI" if pred['is_ai'] else "Human"
                    print(f"  - {model}: {model_pred} ({pred['confidence']:.1%})")
            
            print("-" * 50)
        
    except Exception as e:
        print(f"❌ Text detection demo failed: {e}")


def demo_image_detection():
    """Demonstrate image detection capabilities."""
    print("\n" + "="*50)
    print("🖼️  IMAGE DETECTION DEMO")
    print("="*50)
    
    try:
        # Load image ensemble
        print("Loading image detection models...")
        image_ensemble = ImageEnsemble()
        print("✓ Image models loaded successfully\n")
        
        # Note: In a real demo, you would have actual images
        print("📋 Image Detection Features:")
        print("✓ EfficientNet-based classification")
        print("✓ Frequency domain analysis")
        print("✓ PRNU (Photo Response Non-Uniformity) detection")
        print("✓ Ensemble prediction with confidence scores")
        print("✓ Attention map generation")
        print("✓ Batch processing support")
        
        print("\n📝 Usage Example:")
        print("""
        from PIL import Image
        from src.image_detection import ImageEnsemble
        
        # Load model
        image_ensemble = ImageEnsemble()
        
        # Analyze image
        image = Image.open("path/to/image.jpg")
        result = image_ensemble.predict(image)
        
        print(f"AI Generated: {result['is_ai']}")
        print(f"Confidence: {result['confidence']:.1%}")
        """)
        
        # Get model info
        model_info = image_ensemble.get_model_info()
        print(f"\n🔧 Available Models: {', '.join(model_info['available_models'])}")
        
    except Exception as e:
        print(f"❌ Image detection demo failed: {e}")


def demo_api_usage():
    """Demonstrate API usage."""
    print("\n" + "="*50)
    print("🌐 API USAGE DEMO")
    print("="*50)
    
    print("🚀 Starting the API server:")
    print("uvicorn api.main:app --host 0.0.0.0 --port 8000")
    
    print("\n📋 Available Endpoints:")
    endpoints = [
        ("POST /detect/text", "Detect AI-generated text"),
        ("POST /detect/image", "Detect AI-generated images"),
        ("POST /detect/text/batch", "Batch text detection"),
        ("POST /detect/image/batch", "Batch image detection"),
        ("GET /health", "Health check"),
        ("GET /models/info", "Model information"),
        ("GET /docs", "Interactive API documentation")
    ]
    
    for endpoint, description in endpoints:
        print(f"  {endpoint:<25} - {description}")
    
    print("\n💡 Example cURL commands:")
    
    print("\n# Text detection:")
    print('''curl -X POST "http://localhost:8000/detect/text" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Your text here", "detailed": false}' ''')
    
    print("\n# Image detection:")
    print('''curl -X POST "http://localhost:8000/detect/image" \\
     -F "file=@image.jpg" \\
     -F "detailed=false" ''')
    
    print("\n# Health check:")
    print('curl http://localhost:8000/health')


def demo_streamlit_ui():
    """Demonstrate Streamlit UI."""
    print("\n" + "="*50)
    print("🎨 STREAMLIT UI DEMO")
    print("="*50)
    
    print("🚀 Starting the Streamlit UI:")
    print("streamlit run ui/app.py")
    
    print("\n✨ UI Features:")
    features = [
        "Interactive text analysis with example texts",
        "Image upload and analysis",
        "Batch processing for multiple files",
        "Detailed model explanations and confidence scores",
        "Real-time visualization of results",
        "Model performance metrics",
        "Detection history tracking",
        "Responsive design for desktop and mobile"
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")
    
    print("\n📱 Access the UI at: http://localhost:8501")
    print("🎯 Perfect for interactive testing and demonstrations!")


def main():
    """Run the complete demo."""
    print("🤖 AI CONTENT DETECTION SYSTEM - DEMO")
    print("A hybrid AI-generated text and image detector")
    print("High-accuracy • Low-cost • Production-scale")
    
    # Run demos
    demo_text_detection()
    demo_image_detection() 
    demo_api_usage()
    demo_streamlit_ui()
    
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETE")
    print("="*50)
    print("🚀 Next steps:")
    print("  1. Start the API: uvicorn api.main:app --port 8000")
    print("  2. Launch UI: streamlit run ui/app.py")
    print("  3. Run evaluation: python scripts/evaluate_models.py")
    print("  4. Deploy with Docker: docker-compose up")
    print("\n📚 Check README.md for detailed documentation")
    print("🔗 API docs available at: http://localhost:8000/docs")


if __name__ == "__main__":
    main()