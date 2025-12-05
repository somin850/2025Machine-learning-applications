#!/usr/bin/env python3
"""
Main Search Module - 쿼리 기반 이미지 검색 및 Recall 평가
각 모델별 임베딩 DB에서 쿼리와 유사한 이미지를 검색하고 Recall 성능을 평가
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from tqdm import tqdm
import argparse

# 기본 설정
EMBEDDING_MODEL = "google/embeddinggemma-300m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageSearchEngine:
    """이미지 검색 엔진 클래스"""
    
    def __init__(self, embedding_db_path: str, model_name: str = EMBEDDING_MODEL, device: str = DEVICE):
        """
        ImageSearchEngine 초기화
        
        Args:
            embedding_db_path (str): 임베딩 DB 파일 경로
            model_name (str): 임베딩 모델 이름
            device (str): 사용할 디바이스
        """
        self.embedding_db_path = embedding_db_path
        self.model_name = model_name
        self.device = device
        self.embedding_model = None
        
        # DB 데이터
        self.embeddings = {}
        self.captions = {}
        self.embedding_matrix = None
        self.index_list = []
        
        print(f"🔍 Initializing Image Search Engine")
        print(f"   DB Path: {embedding_db_path}")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        self.load_embedding_db()
        
    def load_embedding_model(self):
        """임베딩 모델 로드"""
        if self.embedding_model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # Mixed precision 설정 (A100 최적화)
            if self.device == "cuda":
                self.embedding_model.half()
                print("   ⚡ Mixed precision enabled")
            
            print("   ✅ Embedding model loaded successfully")
    
    def load_embedding_db(self):
        """임베딩 DB 로드"""
        print(f"📂 Loading embedding DB: {self.embedding_db_path}")
        
        if not os.path.exists(self.embedding_db_path):
            raise FileNotFoundError(f"Embedding DB not found: {self.embedding_db_path}")
        
        with open(self.embedding_db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.embeddings = data.get('embeddings', {})
        self.captions = data.get('captions', {})
        
        if not self.embeddings:
            raise ValueError(f"No embeddings found in {self.embedding_db_path}")
        
        # 임베딩 매트릭스 생성 (빠른 검색을 위해)
        self.index_list = []
        embedding_list = []
        
        for idx_str, embedding in self.embeddings.items():
            self.index_list.append(int(idx_str))
            embedding_list.append(np.array(embedding))
        
        self.embedding_matrix = np.vstack(embedding_list)
        
        print(f"   📊 Loaded {len(self.embeddings)} embeddings")
        print(f"   📐 Embedding dimension: {self.embedding_matrix.shape[1]}")
        
    def embed_query(self, query: str) -> np.ndarray:
        """
        쿼리를 임베딩
        
        Args:
            query (str): 검색 쿼리
            
        Returns:
            np.ndarray: 쿼리 임베딩 벡터
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # EmbeddingGemma는 query embedding을 위한 프롬프트 포맷 사용
        formatted_query = f"Represent this query for retrieving relevant captions: {query}"
        
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    embedding = self.embedding_model.encode(
                        formatted_query,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
            else:
                embedding = self.embedding_model.encode(
                    formatted_query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
        
        return embedding
    
    def search_similar_images(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        쿼리와 유사한 이미지 검색
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 상위 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embed_query(query)
        
        # 코사인 유사도 계산 (이미 정규화되어 있으므로 내적으로 계산)
        similarities = np.dot(self.embedding_matrix, query_embedding)
        
        # 상위 k개 인덱스 찾기
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 결과 구성
        results = []
        for rank, idx in enumerate(top_indices):
            image_index = self.index_list[idx]
            similarity = float(similarities[idx])
            caption = self.captions.get(str(image_index), "No caption available")
            
            result = {
                "rank": rank + 1,
                "image_index": image_index,
                "similarity": similarity,
                "caption": caption,
                "image_path": f"flickr8k_train200/flickr_{image_index:05d}.jpg"  # 이미지 경로 형식
            }
            results.append(result)
        
        return results
    
    def batch_search(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, List[Dict]]:
        """
        배치로 여러 쿼리 검색
        
        Args:
            queries (Dict[str, str]): {query_id: query_text} 형태의 쿼리 딕셔너리
            top_k (int): 각 쿼리당 반환할 상위 결과 수
            
        Returns:
            Dict[str, List[Dict]]: {query_id: search_results} 형태의 결과
        """
        results = {}
        
        print(f"🔍 Searching {len(queries)} queries with top-{top_k} results each")
        
        for query_id, query_text in tqdm(queries.items(), desc="Searching"):
            try:
                search_results = self.search_similar_images(query_text, top_k)
                results[query_id] = search_results
            except Exception as e:
                print(f"❌ Error searching query {query_id}: {e}")
                results[query_id] = []
        
        return results

class RecallEvaluator:
    """Recall 평가 클래스"""
    
    def __init__(self):
        """RecallEvaluator 초기화"""
        pass
    
    def calculate_recall(self, search_results: Dict[str, List[Dict]], 
                        ground_truth: Dict[str, int], 
                        k_values: List[int] = [1, 5, 10]) -> Dict:
        """
        Recall@K 계산
        
        Args:
            search_results (Dict): 검색 결과 {query_id: [results]}
            ground_truth (Dict): 정답 {query_id: correct_image_index}
            k_values (List[int]): 평가할 K 값들
            
        Returns:
            Dict: Recall 결과
        """
        recall_results = {f"recall@{k}": [] for k in k_values}
        detailed_results = {}
        
        for query_id, results in search_results.items():
            if query_id not in ground_truth:
                continue
                
            correct_index = ground_truth[query_id]
            
            # 각 K에 대해 Recall 계산
            query_recalls = {}
            for k in k_values:
                top_k_indices = [result['image_index'] for result in results[:k]]
                is_correct = correct_index in top_k_indices
                recall_results[f"recall@{k}"].append(1.0 if is_correct else 0.0)
                query_recalls[f"recall@{k}"] = 1.0 if is_correct else 0.0
            
            detailed_results[query_id] = {
                "ground_truth": correct_index,
                "top_results": results[:max(k_values)],
                "recalls": query_recalls
            }
        
        # 평균 Recall 계산
        avg_recalls = {}
        for k in k_values:
            recalls = recall_results[f"recall@{k}"]
            avg_recalls[f"recall@{k}"] = np.mean(recalls) if recalls else 0.0
        
        return {
            "average_recalls": avg_recalls,
            "detailed_results": detailed_results,
            "total_queries": len(search_results)
        }

def load_queries(query_file_path: str) -> Dict[str, str]:
    """
    쿼리 파일 로드
    
    Args:
        query_file_path (str): 쿼리 JSON 파일 경로
        
    Returns:
        Dict[str, str]: {query_id: query_text} 형태의 쿼리 딕셔너리
    """
    print(f"📂 Loading queries from: {query_file_path}")
    
    with open(query_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 쿼리 형식이 {"queries": {"0": "query text", ...}} 인 경우
    if 'queries' in data:
        queries = data['queries']
    # 쿼리 형식이 {"0": "query text", ...} 인 경우
    else:
        queries = data
    
    print(f"   📊 Loaded {len(queries)} queries")
    return queries

def create_ground_truth_from_queries(queries: Dict[str, str]) -> Dict[str, int]:
    """
    쿼리 ID를 기반으로 ground truth 생성
    (쿼리 ID가 정답 이미지 인덱스라고 가정)
    
    Args:
        queries (Dict[str, str]): 쿼리 딕셔너리
        
    Returns:
        Dict[str, int]: {query_id: correct_image_index} 형태의 ground truth
    """
    ground_truth = {}
    for query_id in queries.keys():
        try:
            # 쿼리 ID를 정답 이미지 인덱스로 사용
            ground_truth[query_id] = int(query_id)
        except ValueError:
            print(f"⚠️ Warning: Cannot convert query_id '{query_id}' to int")
            continue
    
    return ground_truth

def save_results(results: Dict, output_path: str):
    """
    결과를 JSON 파일로 저장
    
    Args:
        results (Dict): 저장할 결과
        output_path (str): 출력 파일 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {output_path}")

def run_search_evaluation(embedding_db_path: str, query_file_path: str, 
                         output_dir: str, model_type: str, top_k: int = 10):
    """
    검색 및 평가 실행
    
    Args:
        embedding_db_path (str): 임베딩 DB 파일 경로
        query_file_path (str): 쿼리 파일 경로
        output_dir (str): 결과 저장 디렉토리
        model_type (str): 모델 타입 (결과 파일명에 사용)
        top_k (int): 검색할 상위 결과 수
    """
    print(f"\n{'='*20} {model_type} Search Evaluation {'='*20}")
    
    try:
        # 검색 엔진 초기화
        search_engine = ImageSearchEngine(embedding_db_path)
        
        # 쿼리 로드
        queries = load_queries(query_file_path)
        
        # Ground truth 생성 (쿼리 ID = 정답 이미지 인덱스)
        ground_truth = create_ground_truth_from_queries(queries)
        
        # 배치 검색 수행
        search_results = search_engine.batch_search(queries, top_k)
        
        # Recall 평가
        evaluator = RecallEvaluator()
        recall_results = evaluator.calculate_recall(search_results, ground_truth)
        
        # 결과 출력
        print(f"\n📊 {model_type} Recall Results:")
        for metric, value in recall_results['average_recalls'].items():
            print(f"   {metric}: {value:.4f}")
        
        # 전체 결과 구성
        final_results = {
            "model_info": {
                "model_type": model_type,
                "embedding_db_path": embedding_db_path,
                "query_file_path": query_file_path,
                "evaluation_time": datetime.now().isoformat(),
                "top_k": top_k
            },
            "search_results": search_results,
            "recall_evaluation": recall_results
        }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{model_type}_search_results_{timestamp}.json")
        save_results(final_results, output_file)
        
        return final_results
        
    except Exception as e:
        print(f"❌ Error in {model_type} evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Image Search and Recall Evaluation")
    parser.add_argument("--query_file", type=str, required=True, 
                       help="Path to query JSON file")
    parser.add_argument("--output_dir", type=str, default="search_results",
                       help="Output directory for results")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top results to retrieve")
    parser.add_argument("--models", nargs='+', 
                       choices=['blip_base', 'blip_large', 'vit_gpt2', 'vlm', 'vlm_wosimilar', 'all'],
                       default=['all'],
                       help="Models to evaluate")
    
    args = parser.parse_args()
    
    # 모델별 임베딩 DB 경로
    embedding_dbs = {
        'blip_base': "personalized_DB_Embedding/blip_base_embeddings.json",
        'blip_large': "personalized_DB_Embedding/blip_large_embeddings.json",
        'vit_gpt2': "personalized_DB_Embedding/vit_gpt2_embeddings.json",
        'vlm': "personalized_DB_Embedding/VLM_embeddings.json",
        'vlm_wosimilar': "personalized_DB_Embedding/VLM_wosimilar_embeddings.json"
    }
    
    # 평가할 모델 결정
    if 'all' in args.models:
        models_to_evaluate = list(embedding_dbs.keys())
    else:
        models_to_evaluate = args.models
    
    print("=" * 80)
    print("🔍 Image Search and Recall Evaluation")
    print("=" * 80)
    print(f"📂 Query file: {args.query_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔢 Top-K: {args.top_k}")
    print(f"🤖 Models: {', '.join(models_to_evaluate)}")
    print(f"💻 Device: {DEVICE}")
    print()
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"🚀 CUDA Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    all_results = {}
    
    # 각 모델에 대해 평가 수행
    for model_type in models_to_evaluate:
        if model_type not in embedding_dbs:
            print(f"❌ Unknown model type: {model_type}")
            continue
            
        embedding_db_path = embedding_dbs[model_type]
        
        if not os.path.exists(embedding_db_path):
            print(f"❌ Embedding DB not found: {embedding_db_path}")
            continue
        
        results = run_search_evaluation(
            embedding_db_path=embedding_db_path,
            query_file_path=args.query_file,
            output_dir=args.output_dir,
            model_type=model_type,
            top_k=args.top_k
        )
        
        if results:
            all_results[model_type] = results['recall_evaluation']['average_recalls']
    
    # 전체 결과 요약
    if all_results:
        print("\n" + "=" * 80)
        print("📊 Final Recall Comparison")
        print("=" * 80)
        
        # 헤더 출력
        print(f"{'Model':<15} {'Recall@1':<12} {'Recall@5':<12} {'Recall@10':<12}")
        print("-" * 55)
        
        # 각 모델 결과 출력
        for model_type, recalls in all_results.items():
            r1 = recalls.get('recall@1', 0.0)
            r5 = recalls.get('recall@5', 0.0)
            r10 = recalls.get('recall@10', 0.0)
            print(f"{model_type:<15} {r1:<12.4f} {r5:<12.4f} {r10:<12.4f}")
        
        # 종합 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"recall_summary_{timestamp}.json")
        
        summary_data = {
            "evaluation_info": {
                "query_file": args.query_file,
                "top_k": args.top_k,
                "evaluation_time": datetime.now().isoformat(),
                "models_evaluated": list(all_results.keys())
            },
            "recall_results": all_results
        }
        
        save_results(summary_data, summary_file)
        
        print(f"\n🎉 Evaluation completed! Summary saved to: {summary_file}")
    else:
        print("\n❌ No successful evaluations!")

if __name__ == "__main__":
    main()
