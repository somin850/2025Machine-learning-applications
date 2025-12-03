# 📄 config.py
"""
Image Search 프로젝트 설정 파일
"""

import torch
import random
import os

# --- Hugging Face 토큰 설정 ---
# 환경변수에서 토큰을 가져오거나 직접 설정
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', None)
# 또는 직접 토큰을 입력하세요 (보안상 환경변수 사용 권장):


# 토큰이 필요한 모델들
MODELS_REQUIRING_TOKEN = [
    "google/embeddinggemma-300m",
    "HuggingFaceTB/SmolVLM-Instruct"
]

# --- 데이터셋 설정 ---
DATASET_NAME = "Naveengo/flickr8k"
DATASET_SPLIT = "train"

# 실험용 데이터 분리 설정
TOTAL_SAMPLES = None  # None이면 전체 데이터셋 사용
EXPERIMENT_SAMPLES = 400  # 실험용으로 분리할 샘플 수
RANDOM_SEED = 42  # 고정된 시드로 재현 가능한 결과

# 실험 설정
EXPERIMENT_IMAGE_INDEX = 1  # 실험에 사용할 이미지 인덱스 (0~399)
TOP_K_SIMILAR = 5  # 유사한 이미지 상위 K개

# --- 모델 설정 ---
# CLIP 모델 (이미지 임베딩용)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # 더 정확하지만 큰 모델
# CLIP_MODEL_NAME = "google/siglip-base-patch16-224"  # SigLIP (성능 우수)

# VLM 모델 (캡션 생성용) - SmolVLM 사용
VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"  # SmolVLM 2B, 효율적이고 강력한 VLM
# VLM_MODEL_NAME = "Salesforce/blip-image-captioning-large"  # 기존 BLIP 모델
# VLM_MODEL_NAME = "microsoft/git-large-coco"  # 대안 VLM 모델

# Caption Embedding 모델 (텍스트 검색용) - EmbeddingGemma 사용
CAPTION_EMBEDDING_MODEL = "google/embeddinggemma-300m"  # Google EmbeddingGemma 300M (최신, 다국어 지원)
# CAPTION_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # BGE 모델 (성능 우수)
# CAPTION_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 더 가벼운 버전
# CAPTION_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 대안

# --- 생성 옵션 ---
MAX_LENGTH = 50
NUM_BEAMS = 4
TEMPERATURE = 0.7

# --- 파일 경로 설정 ---
# 데이터베이스 파일들 (현재 디렉토리에 저장)
IMAGE_EMBEDDING_DB = "image_embeddings.json"
CAPTION_DB = "captions.json"  # 원본 Flickr8K 캡션만
MY_CAPTION_DB = "my_captions.json"  # 생성된 캡션만
CAPTION_EMBEDDING_DB = "caption_embeddings.json"  # My Caption만 임베딩

# 실험 데이터 파일들 (data 폴더에 저장)
DATA_DIR = "data"
EXPERIMENT_DATA = f"{DATA_DIR}/experiment_data.json"
TRAINING_DATA = f"{DATA_DIR}/training_data.json"

# 결과 파일들 (results 폴더에 저장)
RESULTS_DIR = "results"
SIMILARITY_RESULTS = f"{RESULTS_DIR}/similarity_results.json"
GENERATED_CAPTIONS = f"{RESULTS_DIR}/generated_captions.json"

# --- 장치 설정 ---
# GPU 설정 (A100 80GB MIG3g-40GB 최적화)
FORCE_CPU = False  # True로 설정하면 강제로 CPU 사용
GPU_MEMORY_FRACTION = 0.9  # GPU 메모리 사용 비율 (90%)
MIXED_PRECISION = True  # 혼합 정밀도 사용 (A100에서 성능 향상)

def get_device():
    """사용 가능한 장치 (GPU 또는 CPU)를 반환합니다."""
    if FORCE_CPU:
        print("🖥️  Using CPU (forced)")
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # A100 GPU 최적화 설정
        if "A100" in gpu_name:
            print("⚡ A100 GPU detected - enabling optimizations")
            torch.backends.cudnn.benchmark = True  # A100에서 성능 향상
            if MIXED_PRECISION:
                print("  - Mixed precision enabled")
        
        return device
    else:
        print("🖥️  Using CPU (CUDA not available)")
        return torch.device("cpu")

def setup_gpu_memory():
    """GPU 메모리 설정을 최적화합니다."""
    if torch.cuda.is_available() and not FORCE_CPU:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # 메모리 할당 전략 설정 (A100 최적화)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        print(f"🔧 GPU memory fraction set to {GPU_MEMORY_FRACTION*100}%")

# --- 유틸리티 함수 ---
def set_random_seed(seed: int = None):
    """재현 가능한 결과를 위한 랜덤 시드 설정"""
    if seed is None:
        seed = RANDOM_SEED
    
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 추가적인 재현성을 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_huggingface_token():
    """Hugging Face 토큰을 설정합니다."""
    if HUGGINGFACE_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HUGGINGFACE_TOKEN)
            print("✓ Hugging Face token configured successfully.")
            return True
        except ImportError:
            print("⚠ Warning: huggingface_hub not installed. Installing...")
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "huggingface_hub"])
                from huggingface_hub import login
                login(token=HUGGINGFACE_TOKEN)
                print("✓ Hugging Face token configured successfully.")
                return True
            except Exception as e:
                print(f"❌ Failed to install huggingface_hub: {e}")
                return False
        except Exception as e:
            print(f"❌ Failed to login with Hugging Face token: {e}")
            return False
    else:
        print("⚠ Warning: No Hugging Face token found.")
        print("  Some models may require authentication.")
        print("  Set HUGGINGFACE_TOKEN environment variable or update config.py")
        return False

def check_model_access(model_name: str):
    """모델 접근 권한을 확인합니다."""
    if any(required_model in model_name for required_model in MODELS_REQUIRING_TOKEN):
        if not HUGGINGFACE_TOKEN:
            print(f"⚠ Warning: Model '{model_name}' may require Hugging Face token.")
            print("  Please set your token in config.py or environment variable.")
            return False
    return True
