# 📄 main_wosimilar.py
"""
Image Search 프로젝트 메인 실행 스크립트 (유사 예시 없이)
유사한 이미지 검색 없이 바로 이미지만으로 VLM에 전달하여 캡션을 생성합니다.
CUDA 지원 및 VLM_captions.json 형식 출력
"""

import os
import json
from datetime import datetime
import torch
import config

# OpenMP 오류 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from dataset_loader import load_and_split_dataset
from image_embedder import build_image_embedding_db
from vlm_captioner import create_vlm_captioner
from db_manager import create_database_manager, initialize_databases_from_training_data


def setup_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        config.DATA_DIR,
        config.RESULTS_DIR,
        "personalized_DB"  # VLM 캡션 저장용 디렉토리 추가
    ]
    
    for directory in directories:
        if directory:  # 빈 문자열이 아닌 경우만
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")


def initialize_system():
    """시스템을 초기화합니다."""
    print("=" * 60)
    print("🚀 Image Search System Initialization (Without Similar Examples)")
    print("=" * 60)
    
    # 디렉토리 설정
    setup_directories()
    
    # Hugging Face 토큰 설정
    print("\n--- Step 0: Hugging Face Authentication ---")
    config.setup_huggingface_token()
    
    # 랜덤 시드 설정
    config.set_random_seed()
    
    # GPU 메모리 설정
    config.setup_gpu_memory()
    
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
    
    # 기존 my_captions.json과 caption_embeddings.json 로드 (있는 경우)
    print("\n--- Step 4: Loading Existing Generated Captions ---")
    try:
        # 기존 생성된 캡션 로드
        my_caption_loaded = db_manager.my_caption_db.load_db()
        if my_caption_loaded:
            print(f"✓ Loaded existing my_captions.json: {db_manager.my_caption_db.size()} captions")
        else:
            print("  No existing my_captions.json found - starting fresh")
        
        # 기존 캡션 임베딩 로드
        caption_embedding_loaded = db_manager.caption_embedding_db.load_db()
        if caption_embedding_loaded:
            print(f"✓ Loaded existing caption_embeddings.json: {db_manager.caption_embedding_db.size()} embeddings")
        else:
            print("  No existing caption_embeddings.json found - starting fresh")
            
    except Exception as e:
        print(f"  Warning: Could not load existing files: {e}")
        print("  Starting with empty my_caption and caption_embedding databases")
    
    # 데이터베이스 동기화 확인
    print("\n--- Step 5: Database Synchronization Check ---")
    db_manager.sync_databases()
    
    print("\n✅ System initialization completed successfully!")
    print("=" * 60)
    
    return dataset_loader, db_manager


class VLMCaptionManager:
    """VLM 캡션을 누적 저장하고 관리하는 클래스"""
    
    def __init__(self):
        self.vlm_captions_file = os.path.join("personalized_DB", "VLM_captions_wosimilar.json")
        self.captions = {}
        self.metadata = {}
        self.total_captions = 0
        self.load_existing_captions()
    
    def load_existing_captions(self):
        """기존 VLM 캡션 파일을 로드합니다."""
        if os.path.exists(self.vlm_captions_file):
            try:
                with open(self.vlm_captions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.captions = data.get("captions", {})
                    self.metadata = data.get("metadata", {})
                    self.total_captions = len(self.captions)  # 실제 캡션 개수로 계산
                    print(f"✓ Loaded existing VLM captions: {self.total_captions} entries")
            except Exception as e:
                print(f"  Warning: Could not load existing VLM captions: {e}")
                self.captions = {}
                self.metadata = {}
                self.total_captions = 0
        else:
            print("  No existing VLM captions found - starting fresh")
    
    def add_caption(self, index, caption, metadata):
        """새로운 캡션을 추가합니다."""
        self.captions[str(index)] = caption
        self.metadata[str(index)] = metadata
        self.total_captions = len(self.captions)
    
    def save_captions(self):
        """VLM 캡션을 파일에 저장합니다."""
        vlm_data = {
            "captions": self.captions,
            "metadata": self.metadata,
            "total_captions": self.total_captions
        }
        
        with open(self.vlm_captions_file, 'w', encoding='utf-8') as f:
            json.dump(vlm_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ VLM captions saved: {self.vlm_captions_file} ({self.total_captions} total entries)")
        return self.vlm_captions_file
    
    def get_next_index(self):
        """다음 사용할 인덱스를 반환합니다."""
        if not self.captions:
            return 7691  # VLM_captions.json과 동일한 시작 인덱스
        
        existing_indices = [int(k) for k in self.captions.keys()]
        return max(existing_indices) + 1




def get_simple_prompt():
    """유사 예시 없이 사용할 간단한 프롬프트를 반환합니다."""
    return "Based on these similar image captions above, please generate an accurate and detailed caption for the input image. The caption should be in English"


def run_experiment(dataset_loader, db_manager, vlm_captioner, vlm_caption_manager, experiment_index: int = None):
    """실험을 실행합니다 (유사 예시 없이)."""
    if experiment_index is None:
        experiment_index = config.EXPERIMENT_IMAGE_INDEX
    
    print(f"\n🔍 Running Experiment with Image Index: {experiment_index}")
    print("=" * 60)
    
    # 실험 데이터 가져오기
    experiment_data = dataset_loader.get_experiment_data()
    
    if experiment_index >= len(experiment_data):
        raise ValueError(f"Experiment index {experiment_index} out of range. Max: {len(experiment_data) - 1}")
    
    # 실험 이미지 정보
    query_image = experiment_data[experiment_index]['image']
    query_info = {
        'experiment_index': experiment_index,
        'original_index': experiment_data[experiment_index].get('original_index', experiment_index),
        'caption': experiment_data[experiment_index].get('caption', 'No caption available')
    }
    
    print(f"Query Image Info:")
    print(f"  - Experiment Index: {query_info['experiment_index']}")
    print(f"  - Original Index: {query_info['original_index']}")
    print(f"  - Original Caption: {query_info['caption']}")
    
    # Step 1: VLM으로 캡션 생성 (유사 예시 없이)
    print("\n--- Step 1: Generating Caption with VLM (No Similar Examples) ---")
    simple_prompt = get_simple_prompt()
    
    print(f"Using prompt: {simple_prompt}")
    
    generated_caption = vlm_captioner.generate_caption(
        image=query_image,
        prompt=simple_prompt,
        max_new_tokens=config.MAX_LENGTH,
        temperature=config.TEMPERATURE
    )
    
    print(f"Generated Caption: {generated_caption}")
    
    # Step 2: 새로운 데이터를 DB에 추가
    print("\n--- Step 2: Adding New Data to Databases ---")
    
    # VLM 캡션 매니저에서 다음 인덱스 가져오기
    new_index = vlm_caption_manager.get_next_index()
    
    print(f"  - New VLM caption index: {new_index}")
    
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
        'method': 'without_similar_examples',
        'prompt_used': simple_prompt
    }
    
    # DB에 추가 (기존 방식 유지)
    db_manager.add_new_data(
        index=new_index,
        image_embedding=new_image_embedding.tolist(),
        caption=generated_caption,
        caption_embedding=new_caption_embedding.tolist(),
        metadata=metadata
    )
    
    # VLM 캡션 매니저에 추가
    vlm_caption_manager.add_caption(new_index, generated_caption, metadata)
    
    print(f"✓ New data added with index: {new_index}")
    
    # Step 3: 데이터베이스 저장
    print("\n--- Step 3: Saving Updated Databases ---")
    db_manager.save_all_databases()
    
    # Step 4: VLM 캡션 누적 저장
    print("\n--- Step 4: Saving Accumulated VLM Captions ---")
    vlm_caption_manager.save_captions()
    
    print("\n✅ Experiment completed successfully!")
    print("=" * 60)
    
    return {
        'experiment_index': experiment_index,
        'new_vlm_index': new_index,
        'generated_caption': generated_caption,
        'original_caption': query_info['caption'],
        'success': True
    }


def run_all_experiments(dataset_loader, db_manager, vlm_captioner):
    """0~399까지 모든 실험을 순차적으로 실행합니다."""
    print("\n🔄 Starting Sequential Experiments (0~399) - Without Similar Examples")
    print("=" * 60)
    
    # VLM 캡션 매니저 초기화
    vlm_caption_manager = VLMCaptionManager()
    
    experiment_data = dataset_loader.get_experiment_data()
    total_experiments = len(experiment_data)
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Starting from VLM caption index: {vlm_caption_manager.get_next_index()}")
    
    results_summary = []
    
    for i in range(total_experiments):
        print(f"\n📊 Running Experiment {i+1}/{total_experiments} (Index: {i})")
        print("-" * 50)
        
        try:
            # 각 실험 실행
            experiment_result = run_experiment(dataset_loader, db_manager, vlm_captioner, vlm_caption_manager, experiment_index=i)
            
            # 결과 요약 저장
            summary = {
                'experiment_index': i,
                'success': True,
                'new_vlm_index': experiment_result['new_vlm_index'],
                'generated_caption': experiment_result['generated_caption'],
                'original_caption': experiment_result['original_caption'],
                'timestamp': datetime.now().isoformat()
            }
            results_summary.append(summary)
            
            print(f"✅ Experiment {i} completed (VLM Index: {experiment_result['new_vlm_index']})")
            
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
    
    # 통계 출력
    successful = sum(1 for r in results_summary if r['success'])
    failed = total_experiments - successful
    
    print(f"\n📈 Experiments Summary:")
    print(f"  - Total: {total_experiments}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Success Rate: {successful/total_experiments*100:.1f}%")
    print(f"  - Final VLM captions count: {vlm_caption_manager.total_captions}")
    
    return results_summary


def main():
    """메인 함수"""
    try:
        # 시스템 초기화
        dataset_loader, db_manager = initialize_system()
        
        # VLM 캡셔너 초기화 (한 번만 로드)
        print("\n--- Step 6: Initializing VLM Captioner ---")
        vlm_captioner = create_vlm_captioner()
        print("✓ VLM Captioner initialized")
        
        # 모든 실험 순차 실행
        results_summary = run_all_experiments(dataset_loader, db_manager, vlm_captioner)
        
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