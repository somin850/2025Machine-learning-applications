#!/usr/bin/env python3
"""
빠른 시각화 스크립트 - 특정 쿼리만 빠르게 확인
"""

import sys
import os
import glob
sys.path.append('.')
from visualize_search_results import visualize_all_queries

def find_latest_result_file(model_type: str, search_dir: str = "search_results"):
    """
    특정 모델의 최신 검색 결과 파일 찾기
    
    Args:
        model_type (str): 모델 타입 (blip_base, blip_large, vit_gpt2, vlm, vlm_wosimilar)
        search_dir (str): 검색할 디렉토리
        
    Returns:
        str: 파일 경로 또는 None
    """
    # 모델별 파일명 패턴
    pattern = os.path.join(search_dir, f"{model_type}_search_results_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 최신 파일 반환 (파일명에 타임스탬프가 있으므로 정렬)
    files.sort(reverse=True)
    return files[0]

def list_available_models(search_dir: str = "search_results"):
    """사용 가능한 모델 목록 반환"""
    pattern = os.path.join(search_dir, "*_search_results_*.json")
    files = glob.glob(pattern)
    
    models = set()
    for file in files:
        # 파일명에서 모델 타입 추출: {model}_search_results_{timestamp}.json
        basename = os.path.basename(file)
        model_type = basename.split('_search_results_')[0]
        models.add(model_type)
    
    return sorted(list(models))

def main():
    """빠른 시각화 - 모든 모델에 대해 자동 실행"""
    
    # 사용 가능한 모델 목록
    available_models = list_available_models()
    
    if not available_models:
        print("❌ No search result files found in search_results/")
        print("   Please run main_search.py first to generate search results.")
        return
    
    print("🎨 Quick Visualization - All Models")
    print("=" * 60)
    print(f"\n📋 Found {len(available_models)} models:")
    for model in available_models:
        print(f"   - {model}")
    
    # 쿼리 ID 설정 (0-8번)
    query_ids = [str(i) for i in range(9)]  # 0-8
    
    print(f"\n🔢 Visualizing queries: {', '.join(query_ids)}")
    print("=" * 60)
    print()
    
    # 각 모델에 대해 시각화 실행
    successful_models = []
    failed_models = []
    
    for model in available_models:
        print(f"\n{'='*60}")
        print(f"🤖 Processing Model: {model}")
        print(f"{'='*60}")
        
        # 최신 결과 파일 찾기
        json_file = find_latest_result_file(model)
        
        if not json_file:
            print(f"❌ No search results found for model: {model}")
            failed_models.append(model)
            continue
        
        print(f"📄 Using file: {json_file}")
        print()
        
        try:
            visualize_all_queries(
                json_path=json_file,
                query_ids=query_ids,
                max_queries=None
            )
            successful_models.append(model)
            print(f"✅ {model} visualization completed!")
        except Exception as e:
            print(f"❌ Error visualizing {model}: {e}")
            failed_models.append(model)
            import traceback
            traceback.print_exc()
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📊 Visualization Summary")
    print("=" * 60)
    print(f"✅ Successful: {len(successful_models)} models")
    for model in successful_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"\n❌ Failed: {len(failed_models)} models")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\n📁 Output directory: visualizations/")
    print("=" * 60)

if __name__ == "__main__":
    main()
