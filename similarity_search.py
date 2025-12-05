# 📄 similarity_search.py
"""
코사인 유사도 기반 이미지 검색 모듈
입력 이미지와 가장 유사한 이미지들을 찾는 기능을 제공합니다.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import config


class SimilaritySearcher:
    """코사인 유사도 기반 이미지 검색 클래스"""
    
    def __init__(self, image_embedding_db):
        """
        SimilaritySearcher 초기화
        
        Args:
            image_embedding_db: ImageEmbeddingDB 인스턴스
        """
        self.db = image_embedding_db
        self.db_indices, self.db_embeddings = self.db.get_all_embeddings()
        print(f"SimilaritySearcher initialized with {len(self.db_indices)} embeddings.")
    
    def find_similar_images(self, query_embedding: np.ndarray, top_k: int = None, 
                           similarity_threshold: float = None) -> List[Dict]:
        """
        쿼리 임베딩과 가장 유사한 이미지들을 찾습니다.
        
        Args:
            query_embedding (np.ndarray): 쿼리 이미지의 임베딩 벡터
            top_k (int): 반환할 상위 K개 결과 (기본값: config.TOP_K_SIMILAR)
            similarity_threshold (float): 유사도 임계값 (이 값 이상인 결과만 반환)
        
        Returns:
            List[Dict]: 유사한 이미지들의 정보 리스트
                       각 딕셔너리는 {'index', 'similarity', 'metadata'} 포함
        """
        if top_k is None:
            top_k = config.TOP_K_SIMILAR
        
        # 쿼리 임베딩을 2D 배열로 변환 (cosine_similarity 요구사항)
        query_embedding = query_embedding.reshape(1, -1)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, self.db_embeddings)[0]
        
        # 유사도 순으로 정렬 (내림차순)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 결과 추출 (유사도 임계값 적용)
        results = []
        for i in range(len(sorted_indices)):
            db_idx = sorted_indices[i]
            original_index = self.db_indices[db_idx]
            similarity_score = similarities[db_idx]
            
            # 유사도 임계값 확인
            if similarity_threshold is not None and similarity_score < similarity_threshold:
                break  # 임계값 미만이면 중단 (정렬되어 있으므로 이후는 모두 미만)
            
            # 메타데이터 가져오기
            metadata = self.db.metadata.get(original_index, {})
            
            results.append({
                'index': original_index,
                'similarity': float(similarity_score),
                'metadata': metadata
            })
            
            # top_k 제한 적용 (임계값을 만족하는 결과 중에서)
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_image_index(self, experiment_data: list, experiment_index: int, 
                             top_k: int = None, similarity_threshold: float = None) -> Dict:
        """
        실험 데이터의 특정 이미지로 유사한 이미지들을 검색합니다.
        
        Args:
            experiment_data (list): 실험 데이터 리스트
            experiment_index (int): 실험할 이미지의 인덱스
            top_k (int): 반환할 상위 K개 결과
            similarity_threshold (float): 유사도 임계값
        
        Returns:
            Dict: 검색 결과 {'query_info', 'similar_images', 'filtered_count'}
        """
        if experiment_index >= len(experiment_data):
            raise IndexError(f"Experiment index {experiment_index} out of range")
        
        # 쿼리 이미지 정보
        query_item = experiment_data[experiment_index]
        query_image = query_item['image']
        
        # 쿼리 이미지 임베딩 생성
        from image_embedder import create_image_embedder
        embedder = create_image_embedder()
        query_embedding = embedder.embed_image(query_image)
        
        # 유사한 이미지 검색 (임계값 적용)
        similar_images = self.find_similar_images(query_embedding, top_k, similarity_threshold)
        
        # 결과 구성
        result = {
            'query_info': {
                'experiment_index': experiment_index,
                'original_index': query_item['original_index'],
                'caption': query_item['caption']
            },
            'similar_images': similar_images,
            'filtered_count': len(similar_images),
            'similarity_threshold': similarity_threshold
        }
        
        return result
    
    def get_similar_captions(self, similar_images: List[Dict], db_manager=None) -> List[str]:
        """
        유사한 이미지들의 캡션을 추출합니다.
        원본 캡션 DB와 생성된 캡션 DB 모두에서 검색합니다.
        
        Args:
            similar_images (List[Dict]): 유사한 이미지들의 정보 리스트
            db_manager: DatabaseManager 인스턴스 (캡션 검색용)
        
        Returns:
            List[str]: 캡션 리스트
        """
        captions = []
        
        if db_manager:
            # DB 매니저를 사용하여 캡션 검색 (원본 + 생성된 캡션)
            indices = [img_info['index'] for img_info in similar_images]
            captions = db_manager.get_captions_by_indices(indices)
        else:
            # 기존 방식: 메타데이터에서 캡션 추출
            for img_info in similar_images:
                metadata = img_info.get('metadata', {})
                caption = metadata.get('caption', 'No caption available')
                captions.append(caption)
        
        return captions


def create_similarity_searcher(image_embedding_db):
    """SimilaritySearcher 인스턴스를 생성하는 편의 함수"""
    return SimilaritySearcher(image_embedding_db)


def search_similar_images(experiment_data: list, experiment_index: int, 
                         image_embedding_db, top_k: int = None, 
                         similarity_threshold: float = None) -> Dict:
    """
    실험 이미지로 유사한 이미지들을 검색하는 편의 함수
    
    Args:
        experiment_data (list): 실험 데이터 리스트
        experiment_index (int): 실험할 이미지의 인덱스
        image_embedding_db: ImageEmbeddingDB 인스턴스
        top_k (int): 반환할 상위 K개 결과
        similarity_threshold (float): 유사도 임계값
    
    Returns:
        Dict: 검색 결과
    """
    searcher = create_similarity_searcher(image_embedding_db)
    return searcher.search_by_image_index(experiment_data, experiment_index, top_k, similarity_threshold)
