"""
AI Model Service with Mistral 7B Quantized Model
"""

import torch
import time
import asyncio
from typing import Dict, List, Optional, Any, Generator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from app.core.config import settings
from app.core.monitoring import track_model_inference, record_model_request, update_gpu_utilization, update_memory_usage
import structlog

logger = structlog.get_logger()


class ModelService:
    """Service for managing AI model inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = None
        self.is_loaded = False
        self.model_name = settings.MODEL_NAME
        self.quantization = settings.MODEL_QUANTIZATION
        
        # Model configuration
        self.max_length = settings.MAX_LENGTH
        self.temperature = settings.TEMPERATURE
        self.top_p = settings.TOP_P
        self.top_k = settings.TOP_K
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0
        
    async def load_model(self) -> bool:
        """Load the Mistral 7B model with quantization"""
        try:
            logger.info("Loading Mistral 7B model", model_name=self.model_name, quantization=self.quantization)
            
            # Determine device
            if settings.DEVICE == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = settings.DEVICE
            
            logger.info("Using device", device=self.device)
            
            # Configure quantization
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self.is_loaded = True
            
            # Update memory usage
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                update_memory_usage("gpu", allocated_memory)
                update_gpu_utilization("0", 0.0)  # Will be updated during inference
            
            logger.info("Model loaded successfully", model_name=self.model_name, device=self.device)
            return True
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e), exc_info=True)
            return False
    
    async def generate_text(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the loaded model"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Use default values if not provided
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        
        # Sanitize input
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Record model request
            record_model_request(self.model_name, "started")
            
            # Track inference time
            with track_model_inference(self.model_name, "7B"):
                # Generate text
                if stream:
                    return await self._generate_stream(
                        prompt, max_length, temperature, top_p, top_k
                    )
                else:
                    return await self._generate_batch(
                        prompt, max_length, temperature, top_p, top_k
                    )
                    
        except Exception as e:
            record_model_request(self.model_name, "error")
            logger.error("Text generation failed", error=str(e), exc_info=True)
            raise
    
    async def _generate_batch(
        self, 
        prompt: str, 
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Dict[str, Any]:
        """Generate text in batch mode"""
        
        start_time = time.time()
        
        # Generate text
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        generated_text = outputs[0]['generated_text']
        inference_time = time.time() - start_time
        
        # Update metrics
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Update GPU utilization if using CUDA
        if self.device == "cuda":
            gpu_utilization = torch.cuda.utilization(0)
            update_gpu_utilization("0", gpu_utilization)
            
            allocated_memory = torch.cuda.memory_allocated(0)
            update_memory_usage("gpu", allocated_memory)
        
        record_model_request(self.model_name, "success")
        
        return {
            "text": generated_text,
            "prompt": prompt,
            "inference_time": inference_time,
            "model_name": self.model_name,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        }
    
    async def _generate_stream(
        self, 
        prompt: str, 
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate text in streaming mode"""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate tokens one by one
        generated_tokens = []
        current_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get model outputs
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature and top-k/top-p sampling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Add token to generated sequence
                generated_tokens.append(next_token.item())
                
                # Update inputs for next iteration
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)
                
                # Yield partial result
                partial_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                yield {
                    "text": partial_text,
                    "is_complete": False,
                    "tokens_generated": len(generated_tokens)
                }
        
        # Final result
        full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        # Update metrics
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Update GPU utilization if using CUDA
        if self.device == "cuda":
            gpu_utilization = torch.cuda.utilization(0)
            update_gpu_utilization("0", gpu_utilization)
            
            allocated_memory = torch.cuda.memory_allocated(0)
            update_memory_usage("gpu", allocated_memory)
        
        record_model_request(self.model_name, "success")
        
        yield {
            "text": full_text,
            "prompt": prompt,
            "inference_time": inference_time,
            "is_complete": True,
            "model_name": self.model_name,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "quantization": self.quantization,
            "parameters": {
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            "statistics": {
                "inference_count": self.inference_count,
                "total_inference_time": self.total_inference_time,
                "avg_inference_time": self.total_inference_time / max(self.inference_count, 1)
            }
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        self.is_loaded = False
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")


# Global model service instance
model_service = ModelService()