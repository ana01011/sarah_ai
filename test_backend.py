#!/usr/bin/env python3
"""
Test script for Neural Network Backend
Tests core components without external dependencies
"""
import sys
import os
import asyncio
import traceback

# Add backend to path
sys.path.insert(0, 'backend')

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from app.core.config import settings
        print("✅ Config module imported successfully")
        print(f"   - App name: {settings.APP_NAME}")
        print(f"   - Device: {settings.DEVICE}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from app.models.neural_network import NeuralChatModel
        print("✅ Neural network model imported successfully")
    except Exception as e:
        print(f"❌ Neural network import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from app.services.tokenizer import TokenizerService, AdvancedTokenizer
        print("✅ Tokenizer service imported successfully")
    except Exception as e:
        print(f"❌ Tokenizer import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_neural_network():
    """Test neural network initialization"""
    print("\n🧠 Testing neural network...")
    
    try:
        from app.models.neural_network import NeuralChatModel
        
        # Create a small model for testing
        model = NeuralChatModel(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.1
        )
        
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        
        # Test forward pass
        import torch
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
            print(f"✅ Forward pass successful, output shape: {output['logits'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\n🔤 Testing tokenizer...")
    
    try:
        from app.services.tokenizer import AdvancedTokenizer
        
        tokenizer = AdvancedTokenizer(vocab_size=1000, min_frequency=1)
        
        # Test text preprocessing
        test_text = "Hello, world! This is a test message."
        processed = tokenizer.preprocess_text(test_text)
        print(f"✅ Text preprocessing: '{test_text}' -> '{processed}'")
        
        # Test tokenization
        tokens = tokenizer.tokenize(processed)
        print(f"✅ Tokenization: {len(tokens)} tokens")
        
        # Build small vocabulary
        tokenizer.build_vocabulary([test_text, "Another test sentence."])
        print(f"✅ Vocabulary built: {tokenizer.get_vocab_size()} tokens")
        
        # Test encoding
        encoding = tokenizer.encode(test_text, max_length=20, padding=True)
        print(f"✅ Encoding successful: {len(encoding['input_ids'])} token IDs")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        traceback.print_exc()
        return False

async def test_api_structure():
    """Test API endpoint structure"""
    print("\n🌐 Testing API structure...")
    
    try:
        from app.api.v1 import chat, monitoring
        print("✅ API modules imported successfully")
        
        # Test that routers are created
        if hasattr(chat, 'router'):
            print("✅ Chat router exists")
        else:
            print("❌ Chat router missing")
            return False
            
        if hasattr(monitoring, 'router'):
            print("✅ Monitoring router exists")
        else:
            print("❌ Monitoring router missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration settings"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from app.core.config import settings
        
        # Test required settings
        required_settings = [
            'APP_NAME', 'APP_VERSION', 'API_V1_STR', 'HOST', 'PORT',
            'MODEL_NAME', 'EMBEDDING_DIM', 'NUM_ATTENTION_HEADS'
        ]
        
        for setting in required_settings:
            value = getattr(settings, setting, None)
            if value is not None:
                print(f"✅ {setting}: {value}")
            else:
                print(f"❌ Missing setting: {setting}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🚀 Neural Network Backend Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config),
        ("Neural Network Tests", test_neural_network),
        ("Tokenizer Tests", test_tokenizer),
        ("API Structure Tests", test_api_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)