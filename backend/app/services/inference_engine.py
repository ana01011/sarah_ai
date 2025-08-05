"""
Optimized Inference Engine for Neural Network
Handles model loading, caching, batching, and real-time inference
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from torch.amp import autocast
import redis
import json
import pickle
import psutil
import gc
import os

from ..models.neural_network import NeuralChatModel
from ..services.tokenizer import TokenizerService
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Data class for inference requests"""
    request_id: str
    text: str
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResponse:
    """Data class for inference responses"""
    request_id: str
    generated_text: str
    input_text: str
    processing_time: float
    tokens_generated: int
    model_info: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ModelCache:
    """LRU cache for model states and intermediate results"""
    
    def __init__(self, redis_url: str, max_cache_size: int = 1000):
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        try:
            cached_data = self.redis_client.get(f"inference_cache:{key}")
            if cached_data:
                self.cache_hits += 1
                return pickle.loads(cached_data)
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire_time: int = 3600):
        """Set item in cache with expiration"""
        try:
            serialized_data = pickle.dumps(value)
            self.redis_client.setex(f"inference_cache:{key}", expire_time, serialized_data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete item from cache"""
        try:
            self.redis_client.delete(f"inference_cache:{key}")
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class BatchProcessor:
    """Batch processing for improved throughput"""
    
    def __init__(self, max_batch_size: int = 8, batch_timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[InferenceRequest] = []
        self.request_futures: Dict[str, asyncio.Future] = {}
        self._processing = False
        
    async def add_request(self, request: InferenceRequest) -> InferenceResponse:
        """Add request to batch and return future for response"""
        future = asyncio.Future()
        self.request_futures[request.request_id] = future
        self.pending_requests.append(request)
        
        # Start processing if not already processing
        if not self._processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process pending requests in batches"""
        self._processing = True
        
        try:
            while self.pending_requests:
                # Wait for batch to fill or timeout
                await asyncio.sleep(self.batch_timeout)
                
                if not self.pending_requests:
                    break
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch_requests = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                # Process batch
                await self._execute_batch(batch_requests)
        
        finally:
            self._processing = False
    
    async def _execute_batch(self, batch_requests: List[InferenceRequest]):
        """Execute batch of requests"""
        # This will be called by the inference engine
        pass


class PerformanceMonitor:
    """Monitor inference performance and system resources"""
    
    def __init__(self):
        self.request_count = 0
        self.total_processing_time = 0.0
        self.total_tokens_generated = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def record_request(self, processing_time: float, tokens_generated: int, success: bool = True):
        """Record request metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        self.total_tokens_generated += tokens_generated
        
        if not success:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime = time.time() - self.start_time
        avg_processing_time = (self.total_processing_time / self.request_count 
                             if self.request_count > 0 else 0)
        requests_per_second = self.request_count / uptime if uptime > 0 else 0
        tokens_per_second = self.total_tokens_generated / uptime if uptime > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        # System metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # GPU metrics (if available)
        gpu_metrics = {}
        if torch.cuda.is_available():
            gpu_metrics = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
            }
        
        return {
            "request_count": self.request_count,
            "avg_processing_time": avg_processing_time,
            "requests_per_second": requests_per_second,
            "tokens_per_second": tokens_per_second,
            "error_rate": error_rate,
            "uptime": uptime,
            "memory_usage_percent": memory_info.percent,
            "cpu_usage_percent": cpu_percent,
            **gpu_metrics
        }


class InferenceEngine:
    """
    High-performance inference engine for neural network
    Supports caching, batching, and real-time monitoring
    """
    
    def __init__(self):
        self.model: Optional[NeuralChatModel] = None
        self.tokenizer: Optional[TokenizerService] = None
        self.device = torch.device(settings.DEVICE)
        self.model_loaded = False
        
        # Components
        self.cache = ModelCache(settings.REDIS_URL)
        self.batch_processor = BatchProcessor(max_batch_size=settings.BATCH_SIZE)
        self.performance_monitor = PerformanceMonitor()
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Model optimization settings
        self.use_mixed_precision = settings.MIXED_PRECISION and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        logger.info(f"Initialized InferenceEngine on device: {self.device}")
    
    async def load_model(self, model_path: Optional[str] = None, tokenizer_config: Optional[Dict] = None):
        """Load model and tokenizer"""
        try:
            # Initialize tokenizer
            tokenizer_config = tokenizer_config or {"tokenizer_type": "huggingface", "model_name": "gpt2"}
            self.tokenizer = TokenizerService(**tokenizer_config)
            
            # Initialize model
            vocab_size = self.tokenizer.get_vocab_size()
            self.model = NeuralChatModel(
                vocab_size=vocab_size,
                d_model=settings.EMBEDDING_DIM,
                num_heads=settings.NUM_ATTENTION_HEADS,
                num_layers=settings.NUM_LAYERS,
                d_ff=settings.HIDDEN_DIM,
                max_seq_len=settings.MAX_SEQUENCE_LENGTH,
                dropout=settings.DROPOUT_RATE
            )
            
            # Load pretrained weights if available
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model weights from {model_path}")
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Optimize model for inference
            if hasattr(torch, 'jit') and settings.DEVICE == 'cuda':
                # JIT compilation for better performance
                dummy_input = torch.randint(0, vocab_size, (1, 10), device=self.device)
                try:
                    self.model = torch.jit.trace(self.model, dummy_input)
                    logger.info("Model JIT compiled for better performance")
                except Exception as e:
                    logger.warning(f"JIT compilation failed: {e}")
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def generate_response(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response for a single request"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                logger.debug(f"Cache hit for request {request.request_id}")
                cached_response.request_id = request.request_id
                return cached_response
            
            # Tokenize input
            input_data = self.tokenizer.prepare_inference_data(
                request.text, max_length=settings.MAX_SEQUENCE_LENGTH
            )
            
            input_ids = input_data["input_ids"].to(self.device)
            attention_mask = input_data["attention_mask"].to(self.device)
            
            # Generate response
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast(device_type='cuda'):
                        generated_ids = self.model.generate(
                            input_ids,
                            max_length=request.max_length,
                            temperature=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p,
                            repetition_penalty=request.repetition_penalty,
                            pad_token_id=self.tokenizer.get_special_token_ids().get("pad_token_id"),
                            eos_token_id=self.tokenizer.get_special_token_ids().get("eos_token_id")
                        )
                else:
                    generated_ids = self.model.generate(
                        input_ids,
                        max_length=request.max_length,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                        pad_token_id=self.tokenizer.get_special_token_ids().get("pad_token_id"),
                        eos_token_id=self.tokenizer.get_special_token_ids().get("eos_token_id")
                    )
            
            # Decode response
            generated_text = self.tokenizer.decode_response(generated_ids[0])
            
            # Remove input text from generated text
            if generated_text.startswith(request.text):
                generated_text = generated_text[len(request.text):].strip()
            
            processing_time = time.time() - start_time
            tokens_generated = len(generated_ids[0]) - len(input_ids[0])
            
            # Create response
            response = InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                input_text=request.text,
                processing_time=processing_time,
                tokens_generated=tokens_generated,
                model_info=self.model.get_model_info()
            )
            
            # Cache response
            await self.cache.set(cache_key, response)
            
            # Record metrics
            self.performance_monitor.record_request(processing_time, tokens_generated, True)
            
            logger.debug(f"Generated response for request {request.request_id} in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance_monitor.record_request(processing_time, 0, False)
            logger.error(f"Error generating response for request {request.request_id}: {e}")
            raise
    
    async def generate_batch_responses(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Generate responses for batch of requests"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # For now, process requests individually
        # TODO: Implement true batch processing
        responses = []
        for request in requests:
            try:
                response = await self.generate_response(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in batch processing for request {request.request_id}: {e}")
                # Create error response
                error_response = InferenceResponse(
                    request_id=request.request_id,
                    generated_text="I apologize, but I encountered an error processing your request.",
                    input_text=request.text,
                    processing_time=0.0,
                    tokens_generated=0,
                    model_info={}
                )
                responses.append(error_response)
        
        return responses
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "text": request.text,
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty
        }
        
        import hashlib
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the inference engine"""
        health_status = {
            "status": "healthy",
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            "timestamp": time.time()
        }
        
        try:
            # Test inference with simple request
            if self.model_loaded:
                test_request = InferenceRequest(
                    request_id="health_check",
                    text="Hello",
                    max_length=20
                )
                
                start_time = time.time()
                response = await self.generate_response(test_request)
                health_check_time = time.time() - start_time
                
                health_status.update({
                    "health_check_time": health_check_time,
                    "test_response_length": len(response.generated_text)
                })
            
        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "error": str(e)
            })
        
        return health_status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        performance_metrics = self.performance_monitor.get_metrics()
        cache_stats = self.cache.get_cache_stats()
        
        return {
            "performance": performance_metrics,
            "cache": cache_stats,
            "model_info": self.model.get_model_info() if self.model_loaded else {},
            "system_info": {
                "device": str(self.device),
                "model_loaded": self.model_loaded,
                "mixed_precision": self.use_mixed_precision
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("InferenceEngine cleanup completed")


# Global inference engine instance
inference_engine = InferenceEngine()