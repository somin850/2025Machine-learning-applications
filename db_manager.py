# 📄 db_manager.py
"""
데이터베이스 관리 모듈
이미지 임베딩, 캡션, 캡션 임베딩 데이터베이스를 통합 관리합니다.
"""

import json
import os
from typing import Dict, List, Any
import config


class CaptionDB:
    """캡션 데이터베이스 관리 클래스"""
    
    def __init__(self, db_file: str = None):
        """
        CaptionDB 초기화
        
        Args:
            db_file (str): 캡션 데이터베이스 파일 경로
        """
        if db_file is None:
            db_file = config.CAPTION_DB
        
        self.db_file = db_file
        self.captions = {}  # {index: caption}
        self.metadata = {}  # {index: metadata_dict}
    
    def add_caption(self, index: int, caption: str, metadata: dict = None):
        """
        캡션을 데이터베이스에 추가합니다.
        
        Args:
            index (int): 이미지 인덱스
            caption (str): 캡션 텍스트
            metadata (dict): 메타데이터 (선택사항)
        """
        self.captions[index] = caption
        if metadata:
            self.metadata[index] = metadata
    
    def get_caption(self, index: int) -> str:
        """
        특정 인덱스의 캡션을 가져옵니다.
        
        Args:
            index (int): 이미지 인덱스
        
        Returns:
            str: 캡션 텍스트
        """
        return self.captions.get(index, "")
    
    def get_all_captions(self) -> Dict[int, str]:
        """모든 캡션을 반환합니다."""
        return self.captions.copy()
    
    def update_caption(self, index: int, caption: str, metadata: dict = None):
        """
        기존 캡션을 업데이트합니다.
        
        Args:
            index (int): 이미지 인덱스
            caption (str): 새로운 캡션
            metadata (dict): 새로운 메타데이터
        """
        self.add_caption(index, caption, metadata)
    
    def save_db(self):
        """데이터베이스를 파일로 저장합니다."""
        # 디렉토리가 있는 경우에만 생성
        db_dir = os.path.dirname(self.db_file)
        if db_dir:  # 빈 문자열이 아닌 경우만
            os.makedirs(db_dir, exist_ok=True)
        
        db_data = {
            'captions': self.captions,
            'metadata': self.metadata,
            'total_captions': len(self.captions)
        }
        
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(db_data, f, ensure_ascii=False, indent=2)
            
            print(f"Caption database saved: {self.db_file}")
            print(f"  - Total captions: {len(self.captions)}")
            return True
        except Exception as e:
            print(f"Error saving caption database: {e}")
            return False
    
    def load_db(self):
        """파일에서 데이터베이스를 로드합니다."""
        if not os.path.exists(self.db_file):
            print(f"Caption database file not found: {self.db_file}")
            return False
        
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        self.captions = db_data.get('captions', {})
        self.metadata = db_data.get('metadata', {})
        
        # 키를 정수로 변환
        self.captions = {int(k): v for k, v in self.captions.items()}
        self.metadata = {int(k): v for k, v in self.metadata.items()}
        
        print(f"Caption database loaded: {self.db_file}")
        print(f"  - Total captions: {len(self.captions)}")
        
        return True
    
    def size(self):
        """데이터베이스 크기를 반환합니다."""
        return len(self.captions)


class CaptionEmbeddingDB:
    """캡션 임베딩 데이터베이스 관리 클래스"""
    
    def __init__(self, db_file: str = None):
        """
        CaptionEmbeddingDB 초기화
        
        Args:
            db_file (str): 캡션 임베딩 데이터베이스 파일 경로
        """
        if db_file is None:
            db_file = config.CAPTION_EMBEDDING_DB
        
        self.db_file = db_file
        self.embeddings = {}  # {index: embedding_vector}
        self.metadata = {}    # {index: metadata_dict}
    
    def add_embedding(self, index: int, embedding: list, metadata: dict = None):
        """
        캡션 임베딩을 데이터베이스에 추가합니다.
        
        Args:
            index (int): 이미지 인덱스
            embedding (list): 임베딩 벡터 (리스트 형태)
            metadata (dict): 메타데이터
        """
        self.embeddings[index] = embedding
        if metadata:
            self.metadata[index] = metadata
    
    def get_embedding(self, index: int) -> list:
        """
        특정 인덱스의 임베딩을 가져옵니다.
        
        Args:
            index (int): 이미지 인덱스
        
        Returns:
            list: 임베딩 벡터
        """
        return self.embeddings.get(index, [])
    
    def get_all_embeddings(self) -> tuple:
        """
        모든 임베딩을 반환합니다.
        
        Returns:
            tuple: (indices, embeddings_list)
        """
        indices = list(self.embeddings.keys())
        embeddings_list = [self.embeddings[idx] for idx in indices]
        return indices, embeddings_list
    
    def save_db(self):
        """데이터베이스를 파일로 저장합니다."""
        # 디렉토리가 있는 경우에만 생성
        db_dir = os.path.dirname(self.db_file)
        if db_dir:  # 빈 문자열이 아닌 경우만
            os.makedirs(db_dir, exist_ok=True)
        
        db_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'model_name': config.CAPTION_EMBEDDING_MODEL,
            'total_embeddings': len(self.embeddings),
            'embedding_dim': len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }
        
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(db_data, f, ensure_ascii=False, indent=2)
            
            print(f"Caption embedding database saved: {self.db_file}")
            print(f"  - Total embeddings: {len(self.embeddings)}")
            return True
        except Exception as e:
            print(f"Error saving caption embedding database: {e}")
            return False
    
    def load_db(self):
        """파일에서 데이터베이스를 로드합니다."""
        if not os.path.exists(self.db_file):
            print(f"Caption embedding database file not found: {self.db_file}")
            return False
        
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        self.embeddings = db_data.get('embeddings', {})
        self.metadata = db_data.get('metadata', {})
        
        # 키를 정수로 변환
        self.embeddings = {int(k): v for k, v in self.embeddings.items()}
        self.metadata = {int(k): v for k, v in self.metadata.items()}
        
        print(f"Caption embedding database loaded: {self.db_file}")
        print(f"  - Total embeddings: {len(self.embeddings)}")
        print(f"  - Model used: {db_data.get('model_name', 'Unknown')}")
        print(f"  - Embedding dimension: {db_data.get('embedding_dim', 'Unknown')}")
        
        return True
    
    def size(self):
        """데이터베이스 크기를 반환합니다."""
        return len(self.embeddings)


class DatabaseManager:
    """통합 데이터베이스 관리자"""
    
    def __init__(self):
        """DatabaseManager 초기화"""
        self.image_embedding_db = None
        self.caption_db = CaptionDB()  # 원본 Flickr8K 캡션
        self.my_caption_db = CaptionDB(config.MY_CAPTION_DB)  # 생성된 캡션
        self.caption_embedding_db = CaptionEmbeddingDB()
    
    def set_image_embedding_db(self, image_embedding_db):
        """
        이미지 임베딩 DB를 설정합니다.
        
        Args:
            image_embedding_db: ImageEmbeddingDB 인스턴스
        """
        self.image_embedding_db = image_embedding_db
    
    def add_new_data(self, index: int, image_embedding: list, caption: str, 
                    caption_embedding: list, metadata: dict = None):
        """
        새로운 데이터를 모든 데이터베이스에 추가합니다.
        
        Args:
            index (int): 이미지 인덱스
            image_embedding (list): 이미지 임베딩
            caption (str): 캡션
            caption_embedding (list): 캡션 임베딩
            metadata (dict): 메타데이터
        """
        # 이미지 임베딩 DB에 추가
        if self.image_embedding_db:
            import numpy as np
            self.image_embedding_db.add_embedding(index, np.array(image_embedding), metadata)
        
        # 생성된 캡션은 my_caption_db에 추가
        self.my_caption_db.add_caption(index, caption, metadata)
        
        # 캡션 임베딩 DB에 추가 (생성된 캡션만)
        self.caption_embedding_db.add_embedding(index, caption_embedding, metadata)
    
    def save_all_databases(self):
        """모든 데이터베이스를 저장합니다."""
        print("Saving all databases...")
        
        if self.image_embedding_db:
            self.image_embedding_db.save_db()
        
        self.caption_db.save_db()
        self.caption_embedding_db.save_db()
        
        print("All databases saved successfully.")
    
    def load_all_databases(self):
        """모든 데이터베이스를 로드합니다."""
        print("📂 Loading all databases...")
        
        # 이미지 임베딩 DB 로드
        image_embedding_loaded = True
        if self.image_embedding_db:
            image_embedding_loaded = self.image_embedding_db.load_db()
            print(f"  - Image embeddings: {'✓' if image_embedding_loaded else '❌'}")
        
        # 원본 캡션 DB 로드
        caption_loaded = self.caption_db.load_db()
        print(f"  - Original captions: {'✓' if caption_loaded else '❌'}")
        
        # 생성된 캡션 DB 로드
        my_caption_loaded = self.my_caption_db.load_db()
        print(f"  - My captions: {'✓' if my_caption_loaded else '❌'}")
        
        # 캡션 임베딩 DB 로드 (생성된 캡션만)
        caption_embedding_loaded = self.caption_embedding_db.load_db()
        print(f"  - Caption embeddings: {'✓' if caption_embedding_loaded else '❌'}")
        
        print("Database loading completed.")
        return image_embedding_loaded and caption_loaded and caption_embedding_loaded
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계를 반환합니다."""
        stats = {
            'image_embeddings': self.image_embedding_db.size() if self.image_embedding_db else 0,
            'original_captions': self.caption_db.size(),
            'my_captions': self.my_caption_db.size(),
            'caption_embeddings': self.caption_embedding_db.size()
        }
        return stats
    
    def get_caption_by_index(self, index: int) -> str:
        """
        인덱스로 캡션을 가져옵니다. 
        먼저 my_caption_db에서 찾고, 없으면 original caption_db에서 찾습니다.
        
        Args:
            index (int): 이미지 인덱스
            
        Returns:
            str: 캡션 텍스트
        """
        # 먼저 생성된 캡션에서 찾기
        caption = self.my_caption_db.get_caption(index)
        if caption:
            return caption
        
        # 없으면 원본 캡션에서 찾기
        return self.caption_db.get_caption(index)
    
    def get_captions_by_indices(self, indices: List[int]) -> List[str]:
        """
        여러 인덱스의 캡션들을 가져옵니다.
        
        Args:
            indices (List[int]): 이미지 인덱스 리스트
            
        Returns:
            List[str]: 캡션 텍스트 리스트
        """
        captions = []
        for index in indices:
            caption = self.get_caption_by_index(index)
            if caption:
                captions.append(caption)
        return captions
    
    def save_all_databases(self):
        """모든 데이터베이스를 저장합니다."""
        print("💾 Saving all databases...")
        
        # 이미지 임베딩 DB 저장
        image_embedding_saved = True
        if self.image_embedding_db:
            image_embedding_saved = self.image_embedding_db.save_db()
            print(f"  - Image embeddings: {'✓' if image_embedding_saved else '❌'}")
        
        # 원본 캡션 DB 저장
        caption_saved = self.caption_db.save_db()
        print(f"  - Original captions: {'✓' if caption_saved else '❌'}")
        
        # 생성된 캡션 DB 저장
        my_caption_saved = self.my_caption_db.save_db()
        print(f"  - My captions: {'✓' if my_caption_saved else '❌'}")
        
        # 캡션 임베딩 DB 저장 (생성된 캡션만)
        caption_embedding_saved = self.caption_embedding_db.save_db()
        print(f"  - Caption embeddings: {'✓' if caption_embedding_saved else '❌'}")
        
        return image_embedding_saved and caption_saved and my_caption_saved and caption_embedding_saved
    
    def sync_databases(self):
        """데이터베이스들 간의 동기화를 확인합니다."""
        stats = self.get_database_stats()
        
        print("Database synchronization check:")
        for db_name, count in stats.items():
            print(f"  - {db_name}: {count} entries")
        
        # 모든 DB의 크기가 같은지 확인
        sizes = list(stats.values())
        if len(set(sizes)) == 1:
            print("✓ All databases are synchronized.")
        else:
            print("⚠ Warning: Databases are not synchronized!")
        
        return len(set(sizes)) == 1


def create_caption_db(db_file: str = None):
    """CaptionDB 인스턴스를 생성하는 편의 함수"""
    return CaptionDB(db_file)


def create_caption_embedding_db(db_file: str = None):
    """CaptionEmbeddingDB 인스턴스를 생성하는 편의 함수"""
    return CaptionEmbeddingDB(db_file)


def create_database_manager():
    """DatabaseManager 인스턴스를 생성하는 편의 함수"""
    return DatabaseManager()


def initialize_databases_from_training_data(training_data: list, db_manager: DatabaseManager):
    """
    훈련 데이터로부터 초기 데이터베이스를 구축합니다.
    
    Args:
        training_data (list): 훈련 데이터 리스트
        db_manager (DatabaseManager): 데이터베이스 매니저
    """
    print("Initializing databases from training data...")
    
    # 캡션 DB 초기화
    for item in training_data:
        index = item['training_index']
        caption = item['caption']
        metadata = {'original_index': item['original_index']}
        
        db_manager.caption_db.add_caption(index, caption, metadata)
    
    print(f"Caption database initialized with {len(training_data)} entries.")
    
    # 데이터베이스 저장
    db_manager.caption_db.save_db()
