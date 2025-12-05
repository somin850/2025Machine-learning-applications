#!/usr/bin/env python3
"""
쿼리 JSON 파일을 0번부터 순서대로 정렬하는 스크립트
"""

import json
import sys

def sort_queries_file(input_file: str, output_file: str = None):
    """
    쿼리 JSON 파일을 숫자 순서대로 정렬
    
    Args:
        input_file (str): 입력 파일 경로
        output_file (str): 출력 파일 경로 (None이면 입력 파일에 덮어쓰기)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"📂 Loading: {input_file}")
    
    # JSON 파일 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # queries 객체 가져오기
    queries = data.get('queries', {})
    
    print(f"📊 Found {len(queries)} queries")
    
    # 키를 숫자로 변환하여 정렬
    sorted_queries = {}
    for key in sorted(queries.keys(), key=lambda x: int(x)):
        sorted_queries[key] = queries[key]
    
    # 정렬된 queries로 업데이트
    data['queries'] = sorted_queries
    
    # 파일 저장
    print(f"💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 검증: 0부터 336까지 모두 있는지 확인
    expected_keys = set(str(i) for i in range(337))
    actual_keys = set(sorted_queries.keys())
    
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    
    print(f"\n✅ Sorting completed!")
    print(f"   Total queries: {len(sorted_queries)}")
    
    if missing:
        print(f"   ⚠️  Missing keys: {sorted(list(missing))[:10]}..." if len(missing) > 10 else f"   ⚠️  Missing keys: {sorted(list(missing))}")
    if extra:
        print(f"   ⚠️  Extra keys: {sorted(list(extra))[:10]}..." if len(extra) > 10 else f"   ⚠️  Extra keys: {sorted(list(extra))}")
    
    if not missing and not extra:
        print(f"   ✅ All queries from 0 to 336 are present!")

if __name__ == "__main__":
    input_file = "full_queries_337.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    sort_queries_file(input_file, output_file)
