# 📄 image_embedder.py
"""
CLIP 기반 이미지 임베딩 모듈
이미지를 벡터로 변환하고 DB에 저장/로드하는 기능을 제공합니다.
"""

import json
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import config


class ImageEmbedder:
    """CLIP을 사용하여 이미지를 임베딩하는 클래스"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        ImageEmbedder 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름
            device (str): 사용할 장치
        """
        if model_name is None:
            model_name = config.CLIP_MODEL_NAME
        
        if device is None:
            device = config.get_device()
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading CLIP model: {model_name} on {device}...")
        
        # A100 GPU 최적화를 위한 설정
        if device.type == 'cuda':
            # 혼합 정밀도 지원 확인
            if config.MIXED_PRECISION and torch.cuda.is_available():
                print("  - Loading with mixed precision support")
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        
        # A100에서 최적화
        if device.type == 'cuda':
            self.model.half()  # FP16으로 변환하여 메모리 절약
            print("  - Model converted to FP16 for GPU optimization")
        
        self.model.eval()
        print("CLIP model loaded successfully.")
    
    def embed_image(self, image):
        """
        단일 이미지를 임베딩 벡터로 변환합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
        
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be PIL Image or image path")
        
        # 전처리
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # GPU에서 FP16 사용 시 입력도 변환
        if self.device.type == 'cuda' and config.MIXED_PRECISION:
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].half()
        
        # 임베딩 생성
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # 정규화
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def embed_images_batch(self, images, batch_size: int = 32):
        """
        여러 이미지를 배치로 임베딩합니다.
        
        Args:
            images: PIL Image 객체들의 리스트
            batch_size (int): 배치 크기
        
        Returns:
            numpy.ndarray: 임베딩 벡터들의 배열
        """
        embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch_images = images[i:i+batch_size]
            
            # PIL Image로 변환
            batch_pil = []
            for img in batch_images:
                if isinstance(img, str):
                    batch_pil.append(Image.open(img))
                elif isinstance(img, Image.Image):
                    batch_pil.append(img)
                else:
                    raise ValueError("images must be PIL Images or image paths")
            
            # 전처리
            inputs = self.processor(images=batch_pil, return_tensors="pt", padding=True).to(self.device)
            
            # 임베딩 생성
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)


class ImageEmbeddingDB:
    """이미지 임베딩 데이터베이스 관리 클래스"""
    
    def __init__(self, db_file: str = None):
        """
        ImageEmbeddingDB 초기화
        
        Args:
            db_file (str): 데이터베이스 파일 경로
        """
        if db_file is None:
            db_file = config.IMAGE_EMBEDDING_DB
        
        self.db_file = db_file
        self.embeddings = {}  # {index: embedding_vector}
        self.metadata = {}    # {index: metadata_dict}
    
    def add_embedding(self, index: int, embedding: np.ndarray, metadata: dict = None):
        """
        임베딩을 데이터베이스에 추가합니다.
        
        Args:
            index (int): 이미지 인덱스
            embedding (np.ndarray): 임베딩 벡터
            metadata (dict): 메타데이터 (선택사항)
        """
        self.embeddings[index] = embedding.tolist()  # JSON 저장을 위해 리스트로 변환
        if metadata:
            self.metadata[index] = metadata
    
    def add_embeddings_batch(self, indices: list, embeddings: np.ndarray, metadata_list: list = None):
        """
        여러 임베딩을 배치로 추가합니다.
        
        Args:
            indices (list): 이미지 인덱스들
            embeddings (np.ndarray): 임베딩 벡터들
            metadata_list (list): 메타데이터 리스트 (선택사항)
        """
        for i, index in enumerate(indices):
            embedding = embeddings[i]
            metadata = metadata_list[i] if metadata_list else None
            self.add_embedding(index, embedding, metadata)
    
    def get_embedding(self, index: int):
        """
        특정 인덱스의 임베딩을 가져옵니다.
        
        Args:
            index (int): 이미지 인덱스
        
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        if index not in self.embeddings:
            raise KeyError(f"Index {index} not found in database")
        
        return np.array(self.embeddings[index])
    
    def get_all_embeddings(self):
        """
        모든 임베딩을 반환합니다.
        
        Returns:
            tuple: (indices, embeddings_array)
        """
        indices = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[idx] for idx in indices])
        return indices, embeddings_array
    
    def save_db(self):
        """데이터베이스를 파일로 저장합니다."""
        # 디렉토리가 있는 경우에만 생성
        db_dir = os.path.dirname(self.db_file)
        if db_dir:  # 빈 문자열이 아닌 경우만
            os.makedirs(db_dir, exist_ok=True)
        
        db_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'model_name': config.CLIP_MODEL_NAME,
            'embedding_dim': len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }
        
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(db_data, f, ensure_ascii=False, indent=2)
            
            print(f"Image embedding database saved: {self.db_file}")
            print(f"  - Total embeddings: {len(self.embeddings)}")
            return True
        except Exception as e:
            print(f"Error saving image embedding database: {e}")
            return False
    
    def load_db(self):
        """파일에서 데이터베이스를 로드합니다."""
        if not os.path.exists(self.db_file):
            print(f"Database file not found: {self.db_file}")
            return False
        
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        self.embeddings = db_data.get('embeddings', {})
        self.metadata = db_data.get('metadata', {})
        
        # 키를 정수로 변환 (JSON에서는 문자열로 저장됨)
        self.embeddings = {int(k): v for k, v in self.embeddings.items()}
        self.metadata = {int(k): v for k, v in self.metadata.items()}
        
        print(f"Image embedding database loaded: {self.db_file}")
        print(f"  - Total embeddings: {len(self.embeddings)}")
        print(f"  - Model used: {db_data.get('model_name', 'Unknown')}")
        print(f"  - Embedding dimension: {db_data.get('embedding_dim', 'Unknown')}")
        
        return True
    
    def size(self):
        """데이터베이스 크기를 반환합니다."""
        return len(self.embeddings)


def create_image_embedder(model_name: str = None, device: str = None):
    """ImageEmbedder 인스턴스를 생성하는 편의 함수"""
    return ImageEmbedder(model_name, device)


def create_image_embedding_db(db_file: str = None):
    """ImageEmbeddingDB 인스턴스를 생성하는 편의 함수"""
    return ImageEmbeddingDB(db_file)


def build_image_embedding_db(training_data: list, db_file: str = None, batch_size: int = 32):
    """
    훈련 데이터로부터 이미지 임베딩 데이터베이스를 구축합니다.
    
    Args:
        training_data (list): 훈련 데이터 리스트
        db_file (str): 데이터베이스 파일 경로
        batch_size (int): 배치 크기
    
    Returns:
        ImageEmbeddingDB: 구축된 데이터베이스
    """
    print("Building image embedding database...")
    
    # 임베더와 DB 생성
    embedder = create_image_embedder()
    db = create_image_embedding_db(db_file)
    
    # 기존 DB 로드 시도
    if db.load_db():
        print("Using existing image embedding database.")
        return db
    
    # 이미지들과 인덱스 추출
    images = [item['image'] for item in training_data]
    indices = [item['training_index'] for item in training_data]
    metadata_list = [{'original_index': item['original_index'], 'caption': item['caption']} 
                     for item in training_data]
    
    # 배치로 임베딩 생성
    embeddings = embedder.embed_images_batch(images, batch_size)
    
    # DB에 추가
    db.add_embeddings_batch(indices, embeddings, metadata_list)
    
    # DB 저장
    db.save_db()
    
    print("Image embedding database built successfully.")
    return db
