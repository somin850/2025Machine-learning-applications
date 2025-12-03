# 🔍 Image Search System

유사한 이미지의 캡션을 활용하여 VLM으로 더 나은 캡션을 생성하는 시스템입니다.

## 📋 시스템 개요

이 시스템은 다음과 같은 파이프라인으로 동작합니다:

1. **데이터 준비**: Flickr8K 데이터셋에서 400개를 실험용으로 분리
2. **이미지 임베딩**: CLIP 모델로 훈련 이미지들을 임베딩
3. **유사도 검색**: 입력 이미지와 가장 유사한 5개 이미지 찾기
4. **프롬프트 생성**: 유사한 이미지들의 캡션으로 VLM용 프롬프트 생성
5. **캡션 생성**: VLM으로 새로운 캡션 생성
6. **DB 업데이트**: 생성된 캡션과 임베딩을 데이터베이스에 추가
7. **캡션 임베딩**: BGE 모델로 텍스트 검색용 임베딩 생성

## 🏗️ 프로젝트 구조

```
Image_Search/
├── config.py              # 설정 파일
├── dataset_loader.py       # 데이터셋 로더
├── image_embedder.py       # CLIP 기반 이미지 임베딩
├── similarity_search.py    # 코사인 유사도 검색
├── prompt_generator.py     # VLM용 프롬프트 생성
├── vlm_captioner.py       # VLM 기반 캡션 생성
├── db_manager.py          # 데이터베이스 관리
├── caption_embedder.py    # BGE 기반 캡션 임베딩
├── main.py               # 메인 실행 스크립트
├── requirements.txt      # 의존성 목록
└── README.md            # 이 파일
```

## 🚀 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv image_search_env

# 가상환경 활성화 (Windows)
image_search_env\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source image_search_env/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Hugging Face 토큰 설정 (필수)

일부 모델(EmbeddingGemma, SmolVLM)은 Hugging Face 토큰이 필요합니다:

```bash
# 대화형 토큰 설정
python setup_token.py
```

또는 환경변수로 직접 설정:
```bash
# Windows
set HUGGINGFACE_TOKEN=hf_your_token_here

# macOS/Linux
export HUGGINGFACE_TOKEN=hf_your_token_here
```

**토큰 발급 방법:**
1. [Hugging Face 토큰 페이지](https://huggingface.co/settings/tokens) 방문
2. "New token" 클릭하여 Read 권한으로 토큰 생성
3. 다음 모델 라이선스에 동의:
   - [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)
   - [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### 4. 실행

```bash
python main.py
```

## ⚙️ 설정

`config.py`에서 다음 설정들을 조정할 수 있습니다:

### 데이터셋 설정
- `EXPERIMENT_SAMPLES`: 실험용 샘플 수 (기본값: 400)
- `EXPERIMENT_IMAGE_INDEX`: 실험할 이미지 인덱스 (0~399)
- `RANDOM_SEED`: 재현 가능한 결과를 위한 시드

### 모델 설정
- `CLIP_MODEL_NAME`: 이미지 임베딩용 CLIP 모델
- `VLM_MODEL_NAME`: 캡션 생성용 VLM 모델
- `CAPTION_EMBEDDING_MODEL`: 텍스트 검색용 BGE 모델

### 검색 설정
- `TOP_K_SIMILAR`: 유사한 이미지 상위 K개 (기본값: 5)

## 📊 데이터베이스

시스템은 3개의 데이터베이스를 관리합니다:

1. **Image Embedding DB** (`image_embeddings.json`)
   - 이미지들의 CLIP 임베딩 저장
   - 유사도 검색에 사용

2. **Caption DB** (`captions.json`)
   - 원본 캡션과 생성된 캡션 저장
   - 텍스트 형태로 저장

3. **Caption Embedding DB** (`caption_embeddings.json`)
   - 캡션들의 BGE 임베딩 저장
   - 텍스트 검색에 사용

## 🔧 주요 기능

### 1. 유사도 기반 이미지 검색
- CLIP 임베딩을 사용한 코사인 유사도 계산
- 상위 K개 유사한 이미지 반환

### 2. 컨텍스트 기반 캡션 생성
- 유사한 이미지들의 캡션을 프롬프트로 활용
- VLM으로 더 정확한 캡션 생성

### 3. 동적 데이터베이스 업데이트
- 새로운 캡션 생성 시 자동으로 DB 업데이트
- 지속적인 학습 효과

### 4. 텍스트 검색 지원
- BGE 모델로 캡션 임베딩 생성
- 텍스트 쿼리로 이미지 검색 가능

## 📈 실험 결과

실험 결과는 `results/` 폴더에 저장됩니다:
- `experiment_{index}_{timestamp}.json`: 상세한 실험 결과
- 원본 캡션과 생성된 캡션 비교
- 사용된 유사한 이미지들의 정보

## 🎯 사용 예시

```python
from dataset_loader import load_and_split_dataset
from main import initialize_system, run_experiment

# 시스템 초기화
dataset_loader, db_manager = initialize_system()

# 특정 이미지로 실험 실행
experiment_result = run_experiment(
    dataset_loader, 
    db_manager, 
    experiment_index=42  # 0~399 범위
)

print(f"생성된 캡션: {experiment_result['generated_caption']}")
print(f"원본 캡션: {experiment_result['original_caption']}")
```

## 🔍 모델 추천

### CLIP 모델 (이미지 임베딩)
- `openai/clip-vit-base-patch32`: 기본, 빠름
- `openai/clip-vit-large-patch14`: 더 정확, 큰 모델
- `google/siglip-base-patch16-224`: SigLIP, 성능 우수

### VLM 모델 (캡션 생성)
- `HuggingFaceTB/SmolVLM-Instruct`: 추천 (2B 파라미터, 효율적)
- `Salesforce/blip-image-captioning-large`: 대안
- `microsoft/git-large-coco`: 대안

### 텍스트 임베딩 모델
- `google/embeddinggemma-300m`: 추천 (Google 최신, 다국어 지원)
- `BAAI/bge-large-en-v1.5`: 최고 성능 (영어)
- `BAAI/bge-base-en-v1.5`: 균형잡힌 성능
- `sentence-transformers/all-mpnet-base-v2`: 대안

## 🐛 문제 해결

### 일반적인 오류들

1. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - 더 작은 모델 사용

2. **데이터셋 로드 실패**
   - 인터넷 연결 확인
   - Hugging Face 토큰 설정

3. **모델 로드 실패**
   - PyTorch 버전 확인
   - `safetensors` 설치: `pip install safetensors`

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
