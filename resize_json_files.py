#!/usr/bin/env python3
"""
JSON 파일들을 336개 실험까지만 포함하도록 수정하는 스크립트
"""

import json
import os

def resize_json_file(input_file, output_file, max_index=336):
    """JSON 파일을 지정된 인덱스까지만 포함하도록 수정"""
    
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # captions 필드가 있는 경우
    if 'captions' in data:
        original_count = len(data['captions'])
        
        # VLM 계열 파일인지 확인 (7691번부터 시작하는 경우)
        is_vlm_file = any(key.isdigit() and int(key) >= 7691 for key in data['captions'].keys())
        
        filtered_captions = {}
        filtered_metadata = {}
        
        if is_vlm_file:
            print(f"  VLM 파일 감지: 7691부터 {7691 + max_index}까지 추출")
            # VLM 파일: 7691부터 7691+336까지 (총 337개)
            for i in range(max_index + 1):
                vlm_key = str(7691 + i)  # 7691, 7692, ..., 8027
                new_key = str(i)         # 0, 1, ..., 336
                
                if vlm_key in data['captions']:
                    filtered_captions[new_key] = data['captions'][vlm_key]
                
                # metadata도 같은 방식으로 처리
                if 'metadata' in data and vlm_key in data['metadata']:
                    # metadata의 original_experiment_index를 새로운 키로 업데이트
                    metadata_entry = data['metadata'][vlm_key].copy()
                    if 'original_experiment_index' in metadata_entry:
                        metadata_entry['original_experiment_index'] = i
                    filtered_metadata[new_key] = metadata_entry
        else:
            print(f"  일반 파일 감지: 0부터 {max_index}까지 추출")
            # 일반 파일: 0부터 336까지만 유지 (총 337개)
            for i in range(max_index + 1):
                key = str(i)
                if key in data['captions']:
                    filtered_captions[key] = data['captions'][key]
                
                # metadata도 필터링
                if 'metadata' in data and key in data['metadata']:
                    filtered_metadata[key] = data['metadata'][key]
        
        data['captions'] = filtered_captions
        
        # metadata 업데이트
        if 'metadata' in data:
            data['metadata'] = filtered_metadata
        
        # total_captions 업데이트
        if 'total_captions' in data:
            data['total_captions'] = len(filtered_captions)
        
        print(f"  Original: {original_count} captions")
        print(f"  Filtered: {len(filtered_captions)} captions")
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved to: {output_file}")
    return len(filtered_captions) if 'captions' in data else 0

def main():
    """메인 함수"""
    
    # 파일 목록
    files_to_resize = [
        "personalized_DB/blip_base_captions.json",
        "personalized_DB/blip_large_captions.json", 
        "personalized_DB/vit_gpt2_captions.json",
        "personalized_DB/VLM_captions.json",
        "personalized_DB/VLM_wosimilar_captions.json"
    ]
    
    print("=" * 60)
    print("JSON 파일 크기 조정 (0-336 인덱스까지만 유지)")
    print("=" * 60)
    
    for file_path in files_to_resize:
        if os.path.exists(file_path):
            # 백업 생성
            backup_path = file_path.replace('.json', '_backup.json')
            if not os.path.exists(backup_path):
                with open(file_path, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                print(f"  Backup created: {backup_path}")
            
            # 파일 크기 조정
            count = resize_json_file(file_path, file_path, max_index=336)
            print(f"  ✓ {file_path} - {count} captions")
        else:
            print(f"  ❌ File not found: {file_path}")
        print()
    
    print("✅ JSON 파일 크기 조정 완료!")

if __name__ == "__main__":
    main()
