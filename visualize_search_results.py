#!/usr/bin/env python3
"""
검색 결과 시각화 스크립트
각 쿼리에 대한 상위 10개 이미지를 한 화면에 표시
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datasets import load_dataset
import config
import argparse

def load_search_results(json_path: str):
    """검색 결과 JSON 파일 로드"""
    print(f"📂 Loading search results from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_experiment_mapping(experiment_data_path: str = "data/experiment_data.json"):
    """
    experiment_data.json에서 experiment_index -> original_index 매핑 로드
    
    Args:
        experiment_data_path (str): experiment_data.json 파일 경로
        
    Returns:
        Dict[str, int]: {experiment_index: original_index} 매핑
    """
    if not os.path.exists(experiment_data_path):
        print(f"⚠️  Experiment data not found: {experiment_data_path}")
        return {}
    
    with open(experiment_data_path, 'r', encoding='utf-8') as f:
        experiments = json.load(f)
    
    # experiment_index -> original_index 매핑 생성 (str과 int 키 모두 지원)
    mapping_str = {}
    mapping_int = {}
    for exp in experiments:
        exp_idx = exp.get('experiment_index')
        orig_idx = exp.get('original_index')
        if exp_idx is not None and orig_idx is not None:
            mapping_str[str(exp_idx)] = orig_idx
            mapping_int[exp_idx] = orig_idx
    
    print(f"📊 Loaded {len(mapping_str)} experiment mappings")
    return {'str': mapping_str, 'int': mapping_int}

def get_image_from_dataset(dataset, image_index: int):
    """Hugging Face 데이터셋에서 이미지 가져오기"""
    if image_index < len(dataset):
        sample = dataset[image_index]
        return sample['image']  # PIL Image
    return None

def visualize_query_results(query_id: str, results: list, dataset, output_dir: str = "visualizations", 
                            query_text: str = None, correct_original_index: int = None, 
                            experiment_mapping: dict = None):
    """
    특정 쿼리의 검색 결과를 시각화
    
    Args:
        query_id (str): 쿼리 ID
        results (list): 검색 결과 리스트 (최대 10개)
        dataset: Hugging Face 데이터셋
        output_dir (str): 출력 디렉토리
        query_text (str): 쿼리 텍스트 (제목에 표시)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 최대 10개만 표시
    top_results = results[:10]
    num_images = len(top_results)
    
    if num_images == 0:
        print(f"⚠️  No results for query {query_id}")
        return
    
    # 그리드 설정 (2행 5열)
    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    
    # 제목 생성 (쿼리 텍스트 포함)
    if query_text:
        # 쿼리 텍스트가 너무 길면 자르기 (최대 80자)
        display_text = query_text if len(query_text) <= 80 else query_text[:77] + "..."
        if correct_original_index is not None:
            title = f'Query {query_id} - "{display_text}"\nTop {num_images} Search Results (Correct: original_index={correct_original_index})'
        else:
            title = f'Query {query_id} - "{display_text}"\nTop {num_images} Search Results'
    else:
        if correct_original_index is not None:
            title = f'Query {query_id} - Top {num_images} Search Results (Correct: original_index={correct_original_index})'
        else:
            title = f'Query {query_id} - Top {num_images} Search Results'
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # 각 이미지 표시
    for idx, result in enumerate(top_results):
        row = idx // cols
        col = idx % cols
        
        # axes가 2D 배열인지 1D 배열인지 확인
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col] if isinstance(axes, (list, tuple)) else axes
        
        experiment_index = result['image_index']  # 검색 결과의 image_index는 experiment_index
        similarity = result['similarity']
        caption = result['caption']
        rank = result['rank']
        
        # experiment_index -> original_index 변환
        original_index = None
        if experiment_mapping and isinstance(experiment_mapping, dict):
            # int 키로 먼저 시도, 없으면 str 키로
            if 'int' in experiment_mapping:
                original_index = experiment_mapping['int'].get(experiment_index)
            if original_index is None and 'str' in experiment_mapping:
                original_index = experiment_mapping['str'].get(str(experiment_index))
        
        # original_index가 있으면 그것을 사용, 없으면 experiment_index 그대로 사용
        image_index_to_load = original_index if original_index is not None else experiment_index
        
        # 이미지 가져오기 (original_index 사용)
        pil_image = get_image_from_dataset(dataset, image_index_to_load)
        
        if pil_image:
            ax.imshow(pil_image)
            ax.axis('off')
            
            # 제목: Rank, Similarity, Experiment Index, Original Index
            if original_index is not None:
                title = f"Rank {rank}\nSim: {similarity:.3f}\nExp: {experiment_index}\nOrig: {original_index}"
            else:
                title = f"Rank {rank}\nSim: {similarity:.3f}\nIdx: {experiment_index}"
            ax.set_title(title, fontsize=9, pad=5)
            
            # 정답인지 표시 (검색 결과의 original_index와 Query의 correct_original_index가 같으면)
            is_correct = False
            if correct_original_index is not None and original_index is not None:
                if original_index == correct_original_index:
                    is_correct = True
                    # 녹색 테두리 추가
                    rect = patches.Rectangle((0, 0), pil_image.width-1, pil_image.height-1, 
                                            linewidth=5, edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(10, 20, '✓ CORRECT', fontsize=12, color='green', 
                           weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 정답이지만 rank가 1이 아닌 경우 표시
            if is_correct and rank != 1:
                ax.text(pil_image.width - 100, 20, f'Rank {rank}', fontsize=10, color='orange', 
                       weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Image {image_index}\nNot Found', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # 빈 칸 숨기기
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col] if isinstance(axes, (list, tuple)) else axes
        ax.axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, f"query_{query_id}_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    
    # 표시 (선택적)
    # plt.show()
    plt.close()

def visualize_all_queries(json_path: str, query_ids: list = None, max_queries: int = None):
    """
    모든 쿼리 결과 시각화
    
    Args:
        json_path (str): 검색 결과 JSON 파일 경로
        query_ids (list): 시각화할 쿼리 ID 리스트 (None이면 전체)
        max_queries (int): 최대 시각화할 쿼리 수
    """
    # 검색 결과 로드
    data = load_search_results(json_path)
    search_results = data.get('search_results', {})
    model_type = data.get('model_info', {}).get('model_type', 'unknown')
    query_file_path = data.get('model_info', {}).get('query_file_path', None)
    
    print(f"📊 Found {len(search_results)} queries in results")
    print(f"🤖 Model: {model_type}")
    
    # 쿼리 텍스트 로드
    queries_text = {}
    if query_file_path and os.path.exists(query_file_path):
        print(f"📂 Loading query texts from: {query_file_path}")
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
            queries_text = query_data.get('queries', {})
        print(f"   ✅ Loaded {len(queries_text)} query texts")
    else:
        print(f"   ⚠️  Query file not found, query texts will not be displayed")
    
    # Experiment mapping 로드 (original_index 매핑)
    experiment_mapping = load_experiment_mapping()
    
    # Hugging Face 데이터셋 로드
    print(f"\n📥 Loading Hugging Face dataset: {config.DATASET_NAME}...")
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    print(f"✅ Dataset loaded: {len(dataset)} images")
    
    # 시각화할 쿼리 결정
    if query_ids:
        queries_to_visualize = [qid for qid in query_ids if qid in search_results]
    else:
        queries_to_visualize = list(search_results.keys())
    
    if max_queries:
        queries_to_visualize = queries_to_visualize[:max_queries]
    
    print(f"\n🎨 Visualizing {len(queries_to_visualize)} queries...")
    
    # 출력 디렉토리
    output_dir = f"visualizations/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 쿼리 시각화
    for query_id in queries_to_visualize:
        results = search_results[query_id]
        query_text = queries_text.get(query_id, None)
        
        # experiment_index -> original_index 매핑에서 정답 인덱스 가져오기
        correct_original_index = None
        if experiment_mapping:
            # str 키로 먼저 시도
            correct_original_index = experiment_mapping['str'].get(query_id)
            if correct_original_index is None:
                # int 키로 시도
                try:
                    correct_original_index = experiment_mapping['int'].get(int(query_id))
                except ValueError:
                    pass
        
        if query_text:
            if correct_original_index is not None:
                print(f"  📸 Query {query_id}: \"{query_text[:50]}...\" ({len(results)} results, correct: original_index={correct_original_index})")
            else:
                print(f"  📸 Query {query_id}: \"{query_text[:50]}...\" ({len(results)} results)")
        else:
            if correct_original_index is not None:
                print(f"  📸 Query {query_id}: {len(results)} results (correct: original_index={correct_original_index})")
            else:
                print(f"  📸 Query {query_id}: {len(results)} results")
        
        visualize_query_results(query_id, results, dataset, output_dir, 
                               query_text=query_text, correct_original_index=correct_original_index,
                               experiment_mapping=experiment_mapping)
    
    print(f"\n✅ Visualization completed!")
    print(f"📁 Output directory: {output_dir}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Visualize search results")
    parser.add_argument("--json_file", type=str, required=True,
                       help="Path to search results JSON file")
    parser.add_argument("--queries", nargs='+', type=str, default=None,
                       help="Specific query IDs to visualize (e.g., 0 1 2)")
    parser.add_argument("--max_queries", type=int, default=None,
                       help="Maximum number of queries to visualize")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"❌ File not found: {args.json_file}")
        return
    
    visualize_all_queries(
        json_path=args.json_file,
        query_ids=args.queries,
        max_queries=args.max_queries
    )

if __name__ == "__main__":
    main()
