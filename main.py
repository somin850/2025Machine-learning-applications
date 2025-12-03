# 📄 main.py
"""
Image Search 프로젝트 메인 실행 스크립트
전체 파이프라인을 실행합니다.
"""

import os
import json
from datetime import datetime
import config

# OpenMP 오류 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from dataset_loader import load_and_split_dataset
from image_embedder import build_image_embedding_db
from similarity_search import search_similar_images
from vlm_captioner import generate_caption_with_similarity
from db_manager import create_database_manager, initialize_databases_from_training_data
from caption_embedder import build_caption_embedding_db


def setup_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        config.DATA_DIR,
        config.RESULTS_DIR
    ]
    
    for directory in directories:
        if directory:  # 빈 문자열이 아닌 경우만
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")


def initialize_system():
    """시스템을 초기화합니다."""
    print("=" * 60)
    print("🚀 Image Search System Initialization")
    print("=" * 60)
    
    # 디렉토리 설정
    setup_directories()
    
    # Hugging Face 토큰 설정
    print("\n--- Step 0: Hugging Face Authentication ---")
    config.setup_huggingface_token()
    
    # 랜덤 시드 설정
    config.set_random_seed()
    
    # 데이터셋 로드 및 분리
    print("\n--- Step 1: Loading and Splitting Dataset ---")
    dataset_loader = load_and_split_dataset()
    
    # 훈련 데이터와 실험 데이터 가져오기
    training_data = dataset_loader.get_training_data()
    experiment_data = dataset_loader.get_experiment_data()
    
    print(f"✓ Training data: {len(training_data)} samples")
    print(f"✓ Experiment data: {len(experiment_data)} samples")
    
    # 이미지 임베딩 DB 구축
    print("\n--- Step 2: Building Image Embedding Database ---")
    image_embedding_db = build_image_embedding_db(training_data)
    
    # 데이터베이스 매니저 초기화
    print("\n--- Step 3: Initializing Database Manager ---")
    db_manager = create_database_manager()
    db_manager.set_image_embedding_db(image_embedding_db)
    
    # 캡션 DB 초기화
    initialize_databases_from_training_data(training_data, db_manager)
    
    # 캡션 임베딩 DB는 처음에는 비어있음 (생성된 캡션만 저장)
    print("\n--- Step 4: Initializing Caption Embedding Database ---")
    print("  Caption Embedding DB initialized as empty (will store generated captions only)")
    
    # 데이터베이스 동기화 확인
    print("\n--- Step 5: Database Synchronization Check ---")
    db_manager.sync_databases()
    
    print("\n✅ System initialization completed successfully!")
    print("=" * 60)
    
    return dataset_loader, db_manager


def run_experiment(dataset_loader, db_manager, experiment_index: int = None):
    """실험을 실행합니다."""
    if experiment_index is None:
        experiment_index = config.EXPERIMENT_IMAGE_INDEX
    
    print(f"\n🔍 Running Experiment with Image Index: {experiment_index}")
    print("=" * 60)
    
    # 실험 데이터 가져오기
    experiment_data = dataset_loader.get_experiment_data()
    
    if experiment_index >= len(experiment_data):
        raise ValueError(f"Experiment index {experiment_index} out of range. Max: {len(experiment_data) - 1}")
    
    # Step 1: 유사한 이미지 검색
    print("\n--- Step 1: Finding Similar Images ---")
    search_result = search_similar_images(
        experiment_data=experiment_data,
        experiment_index=experiment_index,
        image_embedding_db=db_manager.image_embedding_db,
        top_k=config.TOP_K_SIMILAR
    )
    
    # 검색 결과 출력
    query_info = search_result['query_info']
    similar_images = search_result['similar_images']
    
    print(f"Query Image Info:")
    print(f"  - Experiment Index: {query_info['experiment_index']}")
    print(f"  - Original Index: {query_info['original_index']}")
    print(f"  - Original Caption: {query_info['caption']}")
    
    print(f"\nTop {len(similar_images)} Similar Images:")
    for i, img_info in enumerate(similar_images, 1):
        print(f"  {i}. Index: {img_info['index']}, Similarity: {img_info['similarity']:.4f}")
        print(f"     Caption: {img_info['metadata']['caption']}")
    
    # Step 2: VLM으로 캡션 생성
    print("\n--- Step 2: Generating Caption with VLM ---")
    query_image = experiment_data[experiment_index]['image']
    
    generated_caption = generate_caption_with_similarity(
        image=query_image,
        search_result=search_result,
        max_new_tokens=config.MAX_LENGTH,
        temperature=config.TEMPERATURE,
        db_manager=db_manager
    )
    
    print(f"Generated Caption: {generated_caption}")
    
    # Step 3: 새로운 데이터를 DB에 추가
    print("\n--- Step 3: Adding New Data to Databases ---")
    
    # 새로운 인덱스 생성 (기존 최대 인덱스 + 1)
    existing_indices = list(db_manager.caption_db.get_all_captions().keys())
    new_index = max(existing_indices) + 1 if existing_indices else 0
    
    # 이미지 임베딩 생성
    from image_embedder import create_image_embedder
    image_embedder = create_image_embedder()
    new_image_embedding = image_embedder.embed_image(query_image)
    
    # 캡션 임베딩 생성 (생성된 캡션만)
    from caption_embedder import create_caption_embedder
    caption_embedder = create_caption_embedder()
    new_caption_embedding = caption_embedder.embed_caption(generated_caption)
    
    # 메타데이터 생성
    metadata = {
        'original_experiment_index': experiment_index,
        'original_index': query_info['original_index'],
        'original_caption': query_info['caption'],
        'generation_timestamp': datetime.now().isoformat(),
        'similar_images_used': [img['index'] for img in similar_images]
    }
    
    # 모든 DB에 추가
    db_manager.add_new_data(
        index=new_index,
        image_embedding=new_image_embedding.tolist(),
        caption=generated_caption,
        caption_embedding=new_caption_embedding.tolist(),
        metadata=metadata
    )
    
    print(f"✓ New data added with index: {new_index}")
    
    # Step 4: 데이터베이스 저장
    print("\n--- Step 4: Saving Updated Databases ---")
    db_manager.save_all_databases()
    
    # Step 5: 결과 저장
    print("\n--- Step 5: Saving Experiment Results ---")
    experiment_result = {
        'experiment_info': {
            'experiment_index': experiment_index,
            'timestamp': datetime.now().isoformat(),
            'new_db_index': new_index
        },
        'query_info': query_info,
        'similar_images': similar_images,
        'generated_caption': generated_caption,
        'original_caption': query_info['caption'],
        'metadata': metadata
    }
    
    # 결과 파일 저장
    result_file = os.path.join(config.RESULTS_DIR, f"experiment_{experiment_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Experiment results saved: {result_file}")
    
    # 데이터베이스 통계 출력
    print("\n--- Final Database Statistics ---")
    stats = db_manager.get_database_stats()
    for db_name, count in stats.items():
        print(f"  - {db_name}: {count} entries")
    
    print("\n✅ Experiment completed successfully!")
    print("=" * 60)
    
    return experiment_result


def run_all_experiments(dataset_loader, db_manager):
    """0~399까지 모든 실험을 순차적으로 실행합니다."""
    print("\n🔄 Starting Sequential Experiments (0~399)")
    print("=" * 60)
    
    experiment_data = dataset_loader.get_experiment_data()
    total_experiments = len(experiment_data)
    
    print(f"Total experiments to run: {total_experiments}")
    
    results_summary = []
    
    for i in range(total_experiments):
        print(f"\n📊 Running Experiment {i+1}/{total_experiments} (Index: {i})")
        print("-" * 50)
        
        try:
            # 각 실험 실행
            experiment_result = run_experiment(dataset_loader, db_manager, experiment_index=i)
            
            # 결과 요약 저장
            summary = {
                'experiment_index': i,
                'success': True,
                'generated_caption': experiment_result['generated_caption'],
                'original_caption': experiment_result['original_caption'],
                'timestamp': experiment_result['experiment_info']['timestamp']
            }
            results_summary.append(summary)
            
            print(f"✅ Experiment {i} completed successfully")
            
            # GPU 메모리 정리 (CUDA 사용 시)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            summary = {
                'experiment_index': i,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results_summary.append(summary)
            continue
    
    # 전체 결과 요약 저장
    summary_file = os.path.join(config.RESULTS_DIR, f"experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    # 통계 출력
    successful = sum(1 for r in results_summary if r['success'])
    failed = total_experiments - successful
    
    print(f"\n📈 Experiments Summary:")
    print(f"  - Total: {total_experiments}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Success Rate: {successful/total_experiments*100:.1f}%")
    print(f"  - Summary saved: {summary_file}")
    
    return results_summary

def main():
    """메인 함수"""
    try:
        # GPU 메모리 설정
        config.setup_gpu_memory()
        
        # 시스템 초기화
        dataset_loader, db_manager = initialize_system()
        
        # 모든 실험 순차 실행
        results_summary = run_all_experiments(dataset_loader, db_manager)
        
        print(f"\n🎉 All experiments completed!")
        
        # 최종 데이터베이스 통계
        print("\n📊 Final Database Statistics:")
        stats = db_manager.get_database_stats()
        for db_name, count in stats.items():
            print(f"  - {db_name}: {count} entries")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
