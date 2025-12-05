#!/usr/bin/env python3
"""
쿼리 생성 실행 스크립트 - Hugging Face 데이터셋 사용
"""

import os
import sys
sys.path.append('.')
from query_maker import QueryGenerator
import config

def main():
    """쿼리 생성 실행 예시"""
    
    print("🤖 Starting Query Generation Process")
    print("=" * 50)
    
    try:
        # 쿼리 생성기 초기화
        generator = QueryGenerator()
        
        # 출력 파일 설정 (VLM 캡션 기반)
        output_file = "VLM_based_queries_337.json"
        
        # 전체 337개 생성
        max_queries = 337
        
        print("🚀 Starting Query Generation Process")
        print(f"📊 Using Hugging Face dataset: {config.DATASET_NAME}")
        print(f"📄 Output: {output_file}")
        print(f"🔢 Max queries: {max_queries}")
        print("-" * 50)
        
        # VLM 캡션 파일 경로 설정
        vlm_captions_path = "personalized_DB/VLM_captions.json"
        
        print(f"📝 Using VLM captions from: {vlm_captions_path}")
        print("-" * 50)
        
        # 쿼리 생성 실행 (resume 모드 - 기존 쿼리 건너뛰기)
        stats = generator.generate_queries_from_dataset(
            dataset_path="data/experiment_data.json",
            image_dir="flickr8k_train200",  # 실제로는 Hugging Face에서 가져올 예정
            output_file=output_file,
            max_queries=max_queries,
            resume=True,  # 기존 쿼리 건너뛰기
            vlm_captions_path=vlm_captions_path  # VLM 캡션 사용
        )
        
        print(f"\n🎉 Query generation completed!")
        print(f"   Total queries: {stats['successful_queries']}/{max_queries}")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Newly generated: {stats['newly_generated']}")
        print(f"   Skipped (existing): {stats['skipped_queries']}")
        print(f"   Failed: {stats['failed_queries']}")
        print(f"   Output file: {output_file}")
        
        # 실패한 쿼리가 있는 경우 재시도 제안
        if stats['failed_queries'] > 0:
            print(f"\n⚠️  {stats['failed_queries']} queries failed to generate.")
            user_input = input("🔄 Retry failed queries? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                print("\n🔄 Retrying failed queries...")
                retry_stats = generator.generate_queries_from_dataset(
                    dataset_path="data/experiment_data.json",
                    image_dir="flickr8k_train200",
                    output_file=output_file,
                    max_queries=max_queries,
                    resume=True,  # 기존 쿼리 건너뛰기
                    vlm_captions_path=vlm_captions_path  # VLM 캡션 사용
                )
                
                print(f"\n🎉 Retry completed!")
                print(f"   Total queries: {retry_stats['successful_queries']}/{max_queries}")
                print(f"   Success rate: {retry_stats['success_rate']:.2%}")
                print(f"   Newly generated: {retry_stats['newly_generated']}")
        
        if stats['successful_queries'] >= max_queries * 0.9:  # 90% 이상 성공
            print("\n✅ Generation successful! Ready for search experiments.")
        else:
            print(f"\n⚠️  Only {stats['success_rate']:.1%} success rate. Consider retrying.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()