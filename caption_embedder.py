# 📄 caption_embedder.py
"""
EmbeddingGemma 기반 캡션 임베딩 모듈
Google의 EmbeddingGemma-300M을 사용하여 캡션을 벡터로 변환하여 텍스트 검색을 지원합니다.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config
from typing import List, Dict, Tuple


class CaptionEmbedder:
    """EmbeddingGemma를 사용하여 캡션을 임베딩하는 클래스"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        CaptionEmbedder 초기화
        
        Args:
            model_name (str): 사용할 임베딩 모델 이름 (EmbeddingGemma)
            device (str): 사용할 장치
        """
        if model_name is None:
            model_name = config.CAPTION_EMBEDDING_MODEL
        
        if device is None:
            device = config.get_device()
        
        self.device = device
        self.model_name = model_name
        
        # 모델 접근 권한 확인 및 토큰 설정
        config.check_model_access(model_name)
        if not config.setup_huggingface_token():
            print("⚠ Proceeding without token - some models may fail to load.")
        
        print(f"Loading EmbeddingGemma model: {model_name} on {device}...")
        try:
            # A100 GPU 최적화 설정
            model_kwargs = {'device': str(device)}
            if device.type == 'cuda' and config.MIXED_PRECISION:
                print("  - Enabling mixed precision for caption embedding")
            
            self.model = SentenceTransformer(model_name, **model_kwargs)
            
            # GPU 메모리 최적화
            if device.type == 'cuda':
                print("  - GPU optimization enabled for caption embedder")
            
            print("EmbeddingGemma model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Possible solutions:")
            print("  1. Set your Hugging Face token in config.py")
            print("  2. Accept the model license at: https://huggingface.co/google/embeddinggemma-300m")
            print("  3. Use alternative model: BAAI/bge-base-en-v1.5")
            raise
    
    def embed_caption(self, caption: str, task_type: str = "retrieval") -> np.ndarray:
        """
        단일 캡션을 임베딩 벡터로 변환합니다.
        EmbeddingGemma의 프롬프트 기반 임베딩을 사용합니다.
        
        Args:
            caption (str): 캡션 텍스트
            task_type (str): 태스크 타입 ("retrieval", "classification", "clustering", "similarity")
        
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        # EmbeddingGemma의 문서 스타일 프롬프트 적용
        if "embeddinggemma" in self.model_name.lower():
            # 문서로 처리 (검색 대상)
            formatted_caption = f"title: none | text: {caption}"
            embedding = self.model.encode_document(formatted_caption, convert_to_numpy=True)
        else:
            # 일반 임베딩 모델
            embedding = self.model.encode(caption, convert_to_numpy=True)
        
        return embedding
    
    def embed_captions_batch(self, captions: List[str], batch_size: int = 32) -> np.ndarray:
        """
        여러 캡션을 배치로 임베딩합니다.
        
        Args:
            captions (List[str]): 캡션 텍스트 리스트
            batch_size (int): 배치 크기
        
        Returns:
            numpy.ndarray: 임베딩 벡터들의 배열
        """
        print(f"Embedding {len(captions)} captions with EmbeddingGemma...")
        
        if "embeddinggemma" in self.model_name.lower():
            # EmbeddingGemma의 문서 스타일 프롬프트 적용
            formatted_captions = [f"title: none | text: {caption}" for caption in captions]
            embeddings = self.model.encode_document(
                formatted_captions,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True
            )
        else:
            # 일반 임베딩 모델
            embeddings = self.model.encode(
                captions, 
                convert_to_numpy=True, 
                batch_size=batch_size,
                show_progress_bar=True
            )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        검색 쿼리를 임베딩합니다.
        EmbeddingGemma의 쿼리 스타일 프롬프트를 사용합니다.
        
        Args:
            query (str): 검색 쿼리
        
        Returns:
            numpy.ndarray: 쿼리 임베딩 벡터
        """
        if "embeddinggemma" in self.model_name.lower():
            # EmbeddingGemma의 쿼리 스타일 프롬프트 적용
            formatted_query = f"task: search result | query: {query}"
            embedding = self.model.encode_query(formatted_query, convert_to_numpy=True)
        else:
            # 일반 임베딩 모델
            embedding = self.model.encode(query, convert_to_numpy=True)
        
        return embedding


class CaptionSearcher:
    """캡션 임베딩 기반 검색 클래스"""
    
    def __init__(self, caption_embedding_db):
        """
        CaptionSearcher 초기화
        
        Args:
            caption_embedding_db: CaptionEmbeddingDB 인스턴스
        """
        self.db = caption_embedding_db
        self.db_indices, self.db_embeddings = self.db.get_all_embeddings()
        
        # 리스트를 numpy 배열로 변환
        if self.db_embeddings:
            self.db_embeddings = np.array(self.db_embeddings)
        else:
            self.db_embeddings = np.array([])
        
        print(f"CaptionSearcher initialized with {len(self.db_indices)} embeddings.")
    
    def search_by_text(self, query_embedding: np.ndarray, top_k: int = 10, 
                      threshold: float = None) -> List[Dict]:
        """
        텍스트 쿼리로 유사한 캡션들을 검색합니다.
        
        Args:
            query_embedding (np.ndarray): 쿼리 임베딩 벡터
            top_k (int): 반환할 상위 K개 결과
            threshold (float): 유사도 임계값
        
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if len(self.db_embeddings) == 0:
            return []
        
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        # 코사인 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.db_embeddings)[0]
        
        # 임계값 이상인 결과만 필터링
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # 유사도 순으로 정렬
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # 상위 K개 결과 추출
        results = []
        for i in range(min(top_k, len(sorted_indices))):
            db_idx = sorted_indices[i]
            original_index = self.db_indices[db_idx]
            similarity_score = similarities[db_idx]
            
            # 메타데이터 가져오기
            metadata = self.db.metadata.get(original_index, {})
            
            results.append({
                'index': original_index,
                'similarity': float(similarity_score),
                'metadata': metadata
            })
        
        return results


def create_caption_embedder(model_name: str = None, device: str = None):
    """CaptionEmbedder 인스턴스를 생성하는 편의 함수"""
    return CaptionEmbedder(model_name, device)


def create_caption_searcher(caption_embedding_db):
    """CaptionSearcher 인스턴스를 생성하는 편의 함수"""
    return CaptionSearcher(caption_embedding_db)


def build_caption_embedding_db(caption_db, caption_embedding_db, batch_size: int = 32):
    """
    캡션 DB로부터 캡션 임베딩 DB를 구축합니다.
    
    Args:
        caption_db: CaptionDB 인스턴스
        caption_embedding_db: CaptionEmbeddingDB 인스턴스
        batch_size (int): 배치 크기
    
    Returns:
        CaptionEmbeddingDB: 구축된 캡션 임베딩 DB
    """
    print("Building caption embedding database...")
    
    # 기존 DB 로드 시도
    if caption_embedding_db.load_db():
        print("Using existing caption embedding database.")
        return caption_embedding_db
    
    # 임베더 생성
    embedder = create_caption_embedder()
    
    # 모든 캡션 가져오기
    all_captions = caption_db.get_all_captions()
    
    if not all_captions:
        print("No captions found in caption database.")
        return caption_embedding_db
    
    # 캡션들과 인덱스 추출
    indices = list(all_captions.keys())
    captions = list(all_captions.values())
    
    # 배치로 임베딩 생성
    embeddings = embedder.embed_captions_batch(captions, batch_size)
    
    # DB에 추가
    for i, index in enumerate(indices):
        embedding = embeddings[i].tolist()  # JSON 저장을 위해 리스트로 변환
        metadata = caption_db.metadata.get(index, {})
        caption_embedding_db.add_embedding(index, embedding, metadata)
    
    # DB 저장
    caption_embedding_db.save_db()
    
    print("Caption embedding database built successfully.")
    return caption_embedding_db


def search_captions_by_text(query: str, caption_embedding_db, 
                           top_k: int = 10, threshold: float = None) -> List[Dict]:
    """
    텍스트 쿼리로 캡션을 검색하는 편의 함수
    
    Args:
        query (str): 검색 쿼리
        caption_embedding_db: CaptionEmbeddingDB 인스턴스
        top_k (int): 반환할 상위 K개 결과
        threshold (float): 유사도 임계값
    
    Returns:
        List[Dict]: 검색 결과
    """
    # 쿼리 임베딩 생성
    embedder = create_caption_embedder()
    query_embedding = embedder.embed_query(query)
    
    # 검색 수행
    searcher = create_caption_searcher(caption_embedding_db)
    results = searcher.search_by_text(query_embedding, top_k, threshold)
    
    return results
