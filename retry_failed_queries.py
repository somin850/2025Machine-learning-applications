#!/usr/bin/env python3
"""
실패한 쿼리만 재생성하는 스크립트
"""

import os
import sys
sys.path.append('.')
from query_maker import QueryGenerator

def main():
    """실패한 쿼리만 재생성"""
    
    input_file = "full_queries_337.json"
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    print("🔄 Retrying Failed Queries")
    print("=" * 50)
    print(f"📂 Input file: {input_file}")
    
    try:
        # 쿼리 생성기 초기화
        generator = QueryGenerator()
        
        # 실패한 쿼리만 재생성 (resume=True)
        stats = generator.generate_queries_from_dataset(
            dataset_path="data/experiment_data.json",
            image_dir="flickr8k_train200",
            output_file=input_file,  # 같은 파일에 덮어쓰기
            max_queries=337,
            resume=True  # 기존 쿼리 건너뛰기
        )
        
        print(f"\n🎉 Retry completed!")
        print(f"   Total queries: {stats['successful_queries']}/337")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Newly generated: {stats['newly_generated']}")
        print(f"   Skipped (existing): {stats['skipped_queries']}")
        print(f"   Failed: {stats['failed_queries']}")
        
        if stats['successful_queries'] >= 320:  # 95% 이상
            print("\n✅ Excellent! Almost all queries generated.")
        elif stats['successful_queries'] >= 300:  # 90% 이상
            print("\n✅ Good! Most queries generated successfully.")
        else:
            print(f"\n⚠️  Still missing {337 - stats['successful_queries']} queries.")
            print("   You may want to run this script again.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
