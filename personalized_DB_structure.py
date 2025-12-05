#!/usr/bin/env python3
"""
Personalized DB Structure - 각 모델별 캡션 임베딩 DB 생성 모듈
각 캡션 생성 모델(BLIP-base, BLIP-large, ViT-GPT2, VLM)의 캡션을 임베딩하여 DB로 저장
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# 기본 설정
EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CaptionEmbeddingDB:
    """캡션 임베딩 데이터베이스 클래스"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        """
        CaptionEmbeddingDB 초기화
        
        Args:
            model_name (str): 임베딩 모델 이름
            device (str): 사용할 디바이스
        """
        self.model_name = model_name
        self.device = device
        self.embedding_model = None
        self.embeddings = {}
        self.captions = {}
        
        print(f"🚀 Initializing Caption Embedding DB")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
    def load_embedding_model(self):
        """임베딩 모델 로드"""
        if self.embedding_model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            
            # EmbeddingGemma 모델 로드
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # Mixed precision 설정 (A100 최적화)
            if self.device == "cuda":
                self.embedding_model.half()
                print("   ⚡ Mixed precision enabled for A100 optimization")
            
            print("   ✅ Embedding model loaded successfully")
    
    def embed_caption(self, caption: str) -> np.ndarray:
        """
        단일 캡션을 임베딩
        
        Args:
            caption (str): 임베딩할 캡션
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # EmbeddingGemma는 document embedding을 위한 프롬프트 포맷 사용
        formatted_caption = f"Represent this caption for retrieval: {caption}"
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    embedding = self.embedding_model.encode(
                        formatted_caption,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
            else:
                embedding = self.embedding_model.encode(
                    formatted_caption,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        
        return embedding
    
    def embed_captions_batch(self, captions: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        배치로 캡션들을 임베딩
        
        Args:
            captions (List[str]): 임베딩할 캡션 리스트
            batch_size (int): 배치 크기
            
        Returns:
            List[np.ndarray]: 임베딩 벡터 리스트
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # 프롬프트 포맷팅
        formatted_captions = [f"Represent this caption for retrieval: {caption}" for caption in captions]
        
        embeddings = []
        
        print(f"🔄 Embedding {len(captions)} captions in batches of {batch_size}")
        
        for i in tqdm(range(0, len(formatted_captions), batch_size), desc="Embedding"):
            batch = formatted_captions[i:i + batch_size]
            
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        batch_embeddings = self.embedding_model.encode(
                            batch,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=batch_size
                        )
                else:
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=batch_size
                    )
            
            embeddings.extend(batch_embeddings)
            
            # GPU 메모리 정리
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return embeddings
    
    def build_db_from_json(self, json_file_path: str, output_file_path: str) -> Dict:
        """
        JSON 파일로부터 임베딩 DB 생성
        
        Args:
            json_file_path (str): 입력 JSON 파일 경로
            output_file_path (str): 출력 임베딩 DB 파일 경로
            
        Returns:
            Dict: 생성된 DB 통계
        """
        print(f"\n📂 Building embedding DB from: {json_file_path}")
        
        # JSON 파일 로드
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        captions_dict = data.get('captions', {})
        
        if not captions_dict:
            raise ValueError(f"No captions found in {json_file_path}")
        
        print(f"   📊 Found {len(captions_dict)} captions")
        
        # 캡션 리스트와 인덱스 준비
        indices = []
        captions = []
        
        for idx_str, caption in captions_dict.items():
            indices.append(int(idx_str))
            captions.append(caption)
            self.captions[int(idx_str)] = caption
        
        # 배치 임베딩 수행
        embeddings = self.embed_captions_batch(captions)
        
        # 임베딩 저장
        for idx, embedding in zip(indices, embeddings):
            self.embeddings[idx] = embedding.tolist()
        
        # 결과 저장
        db_data = {
            "model_info": {
                "embedding_model": self.model_name,
                "device": self.device,
                "total_captions": len(captions)
            },
            "embeddings": self.embeddings,
            "captions": self.captions
        }
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Embedding DB saved to: {output_file_path}")
        
        # 통계 반환
        stats = {
            "total_captions": len(captions),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "model_name": self.model_name,
            "output_file": output_file_path
        }
        
        return stats

def process_all_caption_files():
    """모든 캡션 파일을 처리하여 임베딩 DB 생성"""
    
    # 처리할 파일 목록
    caption_files = [
        {
            "input": "personalized_DB/blip_base_captions.json",
            "output": "personalized_DB_Embedding/blip_base_embeddings.json",
            "name": "BLIP-Base"
        },
        {
            "input": "personalized_DB/blip_large_captions.json", 
            "output": "personalized_DB_Embedding/blip_large_embeddings.json",
            "name": "BLIP-Large"
        },
        {
            "input": "personalized_DB/vit_gpt2_captions.json",
            "output": "personalized_DB_Embedding/vit_gpt2_embeddings.json", 
            "name": "ViT-GPT2"
        },
        {
            "input": "personalized_DB/VLM_captions.json",
            "output": "personalized_DB_Embedding/VLM_embeddings.json",
            "name": "VLM"
        },
        {
            "input": "personalized_DB/VLM_wosimilar_captions.json",
            "output": "personalized_DB_Embedding/VLM_wosimilar_embeddings.json",
            "name": "VLM-WoSimilar"
        }
    ]
    
    print("=" * 80)
    print("🚀 Personalized Caption Embedding DB Generation")
    print("=" * 80)
    print(f"📋 Processing {len(caption_files)} caption files")
    print(f"🤖 Embedding Model: {EMBEDDING_MODEL}")
    print(f"💻 Device: {DEVICE}")
    print()
    
    all_stats = []
    
    for i, file_info in enumerate(caption_files, 1):
        print(f"\n{'='*20} [{i}/{len(caption_files)}] {file_info['name']} {'='*20}")
        
        # 입력 파일 존재 확인
        if not os.path.exists(file_info['input']):
            print(f"❌ Input file not found: {file_info['input']}")
            continue
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(file_info['output']), exist_ok=True)
        
        try:
            # 임베딩 DB 생성
            db = CaptionEmbeddingDB()
            stats = db.build_db_from_json(file_info['input'], file_info['output'])
            stats['model_type'] = file_info['name']
            all_stats.append(stats)
            
            print(f"   📊 Statistics:")
            print(f"      - Total captions: {stats['total_captions']}")
            print(f"      - Embedding dimension: {stats['embedding_dimension']}")
            print(f"      - Output file: {stats['output_file']}")
            
        except Exception as e:
            print(f"❌ Error processing {file_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 전체 통계 출력
    print("\n" + "=" * 80)
    print("📊 Final Statistics")
    print("=" * 80)
    
    for stats in all_stats:
        print(f"✅ {stats['model_type']}")
        print(f"   - Captions: {stats['total_captions']}")
        print(f"   - Dimensions: {stats['embedding_dimension']}")
        print(f"   - File: {os.path.basename(stats['output_file'])}")
        print()
    
    print(f"🎉 Successfully processed {len(all_stats)}/{len(caption_files)} files!")
    
    return all_stats

def main():
    """메인 함수"""
    try:
        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"🚀 CUDA Available: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 Using CPU")
        
        # 모든 캡션 파일 처리
        stats = process_all_caption_files()
        
        if stats:
            print("\n✅ All embedding databases created successfully!")
        else:
            print("\n❌ No files were processed successfully!")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
