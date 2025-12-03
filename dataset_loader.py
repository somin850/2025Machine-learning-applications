# 📄 dataset_loader.py
"""
Flickr8K 데이터셋 로더 모듈
실험용 데이터와 훈련용 데이터를 분리하여 로드합니다.
"""

import json
import os
from datasets import load_dataset
from PIL import Image
import config
import random
from typing import Tuple, List, Dict


class Flickr8KLoader:
    """Flickr8K 데이터셋을 로드하고 실험용/훈련용으로 분리하는 클래스"""
    
    def __init__(self):
        """Flickr8KLoader 초기화"""
        self.dataset = None
        self.experiment_data = []
        self.training_data = []
        
        # 랜덤 시드 설정
        config.set_random_seed()
    
    def load_dataset(self):
        """Flickr8K 데이터셋을 로드합니다."""
        print(f"Loading dataset: {config.DATASET_NAME}...")
        
        try:
            if config.TOTAL_SAMPLES:
                dataset_split = f"{config.DATASET_SPLIT}[:{config.TOTAL_SAMPLES}]"
            else:
                dataset_split = config.DATASET_SPLIT
            
            self.dataset = load_dataset(config.DATASET_NAME, split=dataset_split)
            print(f"Dataset loaded successfully. Total samples: {len(self.dataset)}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def split_data(self):
        """데이터를 실험용과 훈련용으로 분리합니다."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"Splitting data: {config.EXPERIMENT_SAMPLES} for experiment, rest for training...")
        
        # 전체 인덱스 생성
        total_indices = list(range(len(self.dataset)))
        
        # 실험용 인덱스 랜덤 선택 (고정된 시드로)
        experiment_indices = random.sample(total_indices, config.EXPERIMENT_SAMPLES)
        experiment_indices.sort()  # 정렬하여 일관성 유지
        
        # 훈련용 인덱스 (실험용 제외)
        training_indices = [i for i in total_indices if i not in experiment_indices]
        
        # 데이터 분리
        self.experiment_data = []
        self.training_data = []
        
        # 데이터셋 구조 확인
        if len(self.dataset) > 0:
            sample_item = self.dataset[0]
            print(f"Dataset fields: {list(sample_item.keys())}")
            
            # 캡션 필드명 확인 (Flickr8K는 'text' 필드 사용)
            caption_field = None
            possible_caption_fields = ['text', 'caption', 'captions', 'sentence', 'description']
            for field in possible_caption_fields:
                if field in sample_item:
                    caption_field = field
                    break
            
            if caption_field is None:
                raise ValueError(f"No caption field found. Available fields: {list(sample_item.keys())}")
            
            print(f"Using caption field: '{caption_field}'")
        
        for idx in experiment_indices:
            item = self.dataset[idx]
            # 캡션이 리스트인 경우 첫 번째 캡션 사용
            caption = item[caption_field]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            
            self.experiment_data.append({
                'original_index': idx,
                'image': item['image'],
                'caption': caption
            })
        
        for i, idx in enumerate(training_indices):
            item = self.dataset[idx]
            # 캡션이 리스트인 경우 첫 번째 캡션 사용
            caption = item[caption_field]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
                
            self.training_data.append({
                'training_index': i,  # 훈련 데이터 내에서의 인덱스
                'original_index': idx,  # 원본 데이터셋에서의 인덱스
                'image': item['image'],
                'caption': caption
            })
        
        print(f"Data split completed:")
        print(f"  - Experiment data: {len(self.experiment_data)} samples")
        print(f"  - Training data: {len(self.training_data)} samples")
    
    def save_split_data(self):
        """분리된 데이터를 파일로 저장합니다."""
        os.makedirs(os.path.dirname(config.EXPERIMENT_DATA), exist_ok=True)
        os.makedirs(os.path.dirname(config.TRAINING_DATA), exist_ok=True)
        
        # 실험 데이터 저장 (이미지는 경로만 저장)
        experiment_save_data = []
        for item in self.experiment_data:
            experiment_save_data.append({
                'experiment_index': len(experiment_save_data),  # 실험 데이터 내에서의 인덱스
                'original_index': item['original_index'],
                'caption': item['caption']
            })
        
        with open(config.EXPERIMENT_DATA, 'w', encoding='utf-8') as f:
            json.dump(experiment_save_data, f, ensure_ascii=False, indent=2)
        
        # 훈련 데이터 저장 (이미지는 경로만 저장)
        training_save_data = []
        for item in self.training_data:
            training_save_data.append({
                'training_index': item['training_index'],
                'original_index': item['original_index'],
                'caption': item['caption']
            })
        
        with open(config.TRAINING_DATA, 'w', encoding='utf-8') as f:
            json.dump(training_save_data, f, ensure_ascii=False, indent=2)
        
        print(f"Split data saved:")
        print(f"  - Experiment data: {config.EXPERIMENT_DATA}")
        print(f"  - Training data: {config.TRAINING_DATA}")
    
    def load_split_data(self):
        """저장된 분리 데이터를 로드합니다."""
        if os.path.exists(config.EXPERIMENT_DATA) and os.path.exists(config.TRAINING_DATA):
            print("Loading existing split data...")
            
            with open(config.EXPERIMENT_DATA, 'r', encoding='utf-8') as f:
                experiment_save_data = json.load(f)
            
            with open(config.TRAINING_DATA, 'r', encoding='utf-8') as f:
                training_save_data = json.load(f)
            
            # 원본 데이터셋에서 이미지 정보 복원
            self.experiment_data = []
            for item in experiment_save_data:
                original_item = self.dataset[item['original_index']]
                self.experiment_data.append({
                    'experiment_index': item['experiment_index'],
                    'original_index': item['original_index'],
                    'image': original_item['image'],
                    'caption': item['caption']
                })
            
            self.training_data = []
            for item in training_save_data:
                original_item = self.dataset[item['original_index']]
                self.training_data.append({
                    'training_index': item['training_index'],
                    'original_index': item['original_index'],
                    'image': original_item['image'],
                    'caption': item['caption']
                })
            
            print(f"Split data loaded:")
            print(f"  - Experiment data: {len(self.experiment_data)} samples")
            print(f"  - Training data: {len(self.training_data)} samples")
            
            return True
        else:
            print("No existing split data found.")
            return False
    
    def get_experiment_image(self, index: int):
        """실험용 이미지를 가져옵니다."""
        if not self.experiment_data:
            raise ValueError("Experiment data not loaded.")
        
        if index >= len(self.experiment_data):
            raise IndexError(f"Index {index} out of range. Max index: {len(self.experiment_data) - 1}")
        
        return self.experiment_data[index]
    
    def get_training_data(self):
        """훈련용 데이터를 반환합니다."""
        return self.training_data
    
    def get_experiment_data(self):
        """실험용 데이터를 반환합니다."""
        return self.experiment_data


def create_dataset_loader():
    """DatasetLoader 인스턴스를 생성하는 편의 함수"""
    return Flickr8KLoader()


def load_and_split_dataset():
    """데이터셋을 로드하고 분리하는 전체 프로세스"""
    loader = create_dataset_loader()
    
    # 데이터셋 로드
    loader.load_dataset()
    
    # 기존 분리 데이터가 있는지 확인
    if not loader.load_split_data():
        # 없으면 새로 분리
        loader.split_data()
        loader.save_split_data()
    
    return loader
