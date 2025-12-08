# 🔍 Image Search System

유사한 이미지의 캡션을 활용하여 VLM으로 더 나은 캡션을 생성하고, 이를 기반으로 이미지 검색을 수행하는 시스템입니다.

---

## 📋 빠른 시작 가이드

### 0️⃣ 준비사항
- Python 3.10 이상
- GPU 권장 (CPU도 가능하지만 느림)

### 1️⃣ 가상환경 설정 및 패키지 설치

```powershell
# 가상환경 생성
python -m venv image_search_env

# 가상환경 활성화 (PowerShell)
image_search_env\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ Hugging Face 토큰 설정

`config.py` 파일을 열어서 `HUGGINGFACE_TOKEN` 주석을 제거하고 본인의 토큰을 입력하세요.

```python
# config.py
HUGGINGFACE_TOKEN = "hf_your_token_here"  # 주석 제거 후 토큰 입력
```

**토큰 발급 방법:**
1. [Hugging Face 토큰 페이지](https://huggingface.co/settings/tokens) 방문
2. "New token" 클릭하여 Read 권한으로 토큰 생성
3. 다음 모델 라이선스에 동의:
   - [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)
   - [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### 3️⃣ VLM 캡션 생성

```powershell
python main.py
```

**실행 후:**
- 생성된 출력 파일을 `VLM_captions.json`으로 이름 변경
- `personalized_DB` 폴더 생성 (없는 경우)
- `VLM_captions.json`을 `personalized_DB` 폴더 안으로 이동

### 4️⃣ JSON 파일 정렬 및 동기화

```powershell
# 1단계: 백업 파일 기반으로 정렬
python resize_json_files.py

# 확인: personalized_DB 폴더에 다음 두 파일이 있는지 확인
# - VLM_captions.json
# - VLM_captions_backup.json

# 2단계: 모든 파일 동기화
python sync_all_captions.py
```

### 5️⃣ 임베딩 생성

```powershell
python personalized_DB_structure.py
```

**확인:** `personalized_DB_Embedding` 폴더 안에 `VLM_embeddings.json` 파일이 생성되었는지 확인

### 6️⃣ 이미지 검색 실행

```powershell
# 짧은 쿼리로 검색
python main_search.py --query_file Query_short.json

# 상세 쿼리로 검색
python main_search.py --query_file Query_detail.json
```

**결과:** 검색 결과는 콘솔에 출력되며, `search_results/` 폴더에도 저장됩니다.

---

## 🎯 전체 워크플로우

### A. 캡션 생성 단계

#### A-1. VLM 캡션 생성 (유사 이미지 활용)
```powershell
python main.py
```
- 유사한 이미지 5개의 캡션을 프롬프트로 활용하여 VLM으로 캡션 생성
- 출력: `VLM_captions.json` → `personalized_DB/VLM_captions.json`으로 이동

#### A-2. VLM 캡션 생성 (유사 이미지 없이)
```powershell
python main_wosimilar_2.py
```
- 유사 이미지 검색 없이 바로 VLM으로 캡션 생성
- 출력: `VLM_wosimilar_captions.json` → `personalized_DB/VLM_wosimilar_captions.json`으로 이동

#### A-3. BLIP Base 캡션 생성
```powershell
python blip_base_captioner.py
```
- BLIP Base 모델로 캡션 생성
- 출력: `blip_base_captions.json` → `personalized_DB/blip_base_captions.json`으로 이동

#### A-4. BLIP Large 캡션 생성
```powershell
python blip_large_captioner.py
```
- BLIP Large 모델로 캡션 생성
- 출력: `blip_large_captions.json` → `personalized_DB/blip_large_captions.json`으로 이동

#### A-5. ViT-GPT2 캡션 생성
```powershell
python vit_gpt2_captioner.py
```
- ViT-GPT2 모델로 캡션 생성
- 출력: `vit_gpt2_captions.json` → `personalized_DB/vit_gpt2_captions.json`으로 이동

### B. 데이터 정리 단계

#### B-1. JSON 파일 정렬
```powershell
python resize_json_files.py
```
- 모든 백업 파일(`*_backup.json`)을 읽어서 정렬된 JSON 파일 생성
- `experiment_index`를 기준으로 모든 파일을 동일한 인덱스로 매핑
- 누락된 인덱스는 다른 파일에서도 제거

#### B-2. 파일 동기화
```powershell
python sync_all_captions.py
```
- `VLM_captions.json`을 기준으로 모든 캡션 파일과 쿼리 파일 동기화
- 동일한 `experiment_index`만 유지

### C. 임베딩 생성 단계

```powershell
python personalized_DB_structure.py
```
- 모든 캡션 파일을 읽어서 텍스트 임베딩 생성
- `personalized_DB_Embedding/` 폴더에 각 모델별 임베딩 파일 저장

### D. 쿼리 생성 단계

#### D-1. 쿼리 생성 (GPT-4o-mini 사용)
```powershell
python run_query_generation.py
```
또는 직접 실행:
```powershell
python query_maker.py --output Query_short.json --query_type short
python query_maker.py --output Query_detail.json --query_type detail
```

**Query Maker 설명:**
- OpenAI GPT-4o-mini API를 사용하여 이미지 검색 쿼리 생성
- 이미지와 원본 캡션을 기반으로 인간이 실제 검색할 때 사용할 법한 쿼리 생성
- `short`: 짧은 쿼리 (예: "dog playing")
- `detail`: 상세 쿼리 (예: "a brown dog playing with a red ball in the park")
- VLM 캡션을 사용하여 더 정확한 쿼리 생성 가능

#### D-2. 쿼리 정렬
```powershell
python sort_queries.py
```
- 쿼리 파일을 `experiment_index` 순서로 정렬

### E. 검색 실행 단계

```powershell
python main_search.py --query_file Query_short.json
python main_search.py --query_file Query_detail.json
```

**검색 프로세스:**
1. 쿼리 파일 로드
2. 각 모델별 캡션 임베딩과 쿼리 임베딩 비교
3. 코사인 유사도 기반으로 상위 K개 이미지 검색
4. Recall@K 계산 및 결과 저장

---

## 📁 프로젝트 파일 구조 및 설명

### 🎬 메인 실행 스크립트

| 파일명 | 설명 |
|--------|------|
| `main.py` | **VLM 캡션 생성 (유사 이미지 활용)** - 유사한 이미지 5개의 캡션을 프롬프트로 사용하여 VLM으로 캡션 생성 |
| `main_wosimilar_2.py` | **VLM 캡션 생성 (유사 이미지 없이)** - 유사 이미지 검색 없이 바로 VLM으로 캡션 생성 |
| `main_search.py` | **이미지 검색 실행** - 쿼리 파일을 읽어서 각 모델별로 이미지 검색 수행 및 Recall 계산 |
| `blip_base_captioner.py` | **BLIP Base 캡션 생성** - BLIP Base 모델로 실험 이미지들의 캡션 생성 |
| `blip_large_captioner.py` | **BLIP Large 캡션 생성** - BLIP Large 모델로 실험 이미지들의 캡션 생성 |
| `vit_gpt2_captioner.py` | **ViT-GPT2 캡션 생성** - ViT-GPT2 모델로 실험 이미지들의 캡션 생성 |

### 🔧 데이터 처리 스크립트

| 파일명 | 설명 |
|--------|------|
| `resize_json_files.py` | **JSON 파일 정렬** - 백업 파일을 읽어서 `experiment_index` 기준으로 정렬된 JSON 파일 생성 |
| `sync_all_captions.py` | **파일 동기화** - `VLM_captions.json`을 기준으로 모든 캡션/쿼리 파일의 인덱스 동기화 |
| `personalized_DB_structure.py` | **임베딩 생성** - 모든 캡션 파일을 읽어서 텍스트 임베딩 생성 및 저장 |
| `query_maker.py` | **쿼리 생성기** - GPT-4o-mini를 사용하여 이미지 검색 쿼리 생성 |
| `run_query_generation.py` | **쿼리 생성 실행** - `query_maker.py`를 실행하여 쿼리 파일 생성 |
| `sort_queries.py` | **쿼리 정렬** - 쿼리 파일을 `experiment_index` 순서로 정렬 |

### 🎨 시각화 및 분석

| 파일명 | 설명 |
|--------|------|
| `quick_visualize.py` | **빠른 시각화** - 검색 결과를 빠르게 시각화 |
| `visualize_search_results.py` | **검색 결과 시각화** - 상세한 검색 결과 시각화 및 저장 |

### 🛠️ 핵심 모듈

| 파일명 | 설명 |
|--------|------|
| `config.py` | **설정 파일** - 모든 하이퍼파라미터, 모델 경로, 디렉토리 경로 등 설정 |
| `dataset_loader.py` | **데이터셋 로더** - Flickr8K 데이터셋 로드 및 훈련/실험 데이터 분리 |
| `image_embedder.py` | **이미지 임베딩** - CLIP 모델을 사용하여 이미지 임베딩 생성 |
| `caption_embedder.py` | **캡션 임베딩** - BGE 모델을 사용하여 텍스트 캡션 임베딩 생성 |
| `similarity_search.py` | **유사도 검색** - 코사인 유사도를 사용한 이미지/텍스트 검색 |
| `prompt_generator.py` | **프롬프트 생성** - 유사한 이미지들의 캡션을 기반으로 VLM용 프롬프트 생성 |
| `vlm_captioner.py` | **VLM 캡션 생성기** - VLM 모델을 사용하여 이미지 캡션 생성 |
| `db_manager.py` | **데이터베이스 관리** - 이미지 임베딩, 캡션, 캡션 임베딩 DB 관리 |

### 🔍 유틸리티 스크립트

| 파일명 | 설명 |
|--------|------|
| `setup_token.py` | **토큰 설정** - Hugging Face 토큰을 대화형으로 설정 |
| `retry_failed_queries.py` | **실패한 쿼리 재시도** - 실패한 쿼리 생성 작업 재시도 |

---

## 📂 주요 디렉토리 구조

```
Image_Search/
├── personalized_DB/              # 캡션 파일 저장
│   ├── VLM_captions.json
│   ├── VLM_captions_backup.json
│   ├── blip_base_captions.json
│   ├── blip_large_captions.json
│   └── ...
├── personalized_DB_Embedding/    # 임베딩 파일 저장
│   ├── VLM_embeddings.json
│   ├── blip_base_embeddings.json
│   └── ...
├── search_results/                # 검색 결과 저장
├── results/                       # 실험 결과 저장
├── visualizations/               # 시각화 결과 저장
└── data/                         # 데이터셋 파일
    ├── training_data.json
    └── experiment_data.json
```

---

## ⚙️ 설정 파일 (`config.py`)

주요 설정 항목:
- `HUGGINGFACE_TOKEN`: Hugging Face API 토큰 (필수)
- `EXPERIMENT_SAMPLES`: 실험용 샘플 수 (기본값: 400)
- `CLIP_MODEL_NAME`: 이미지 임베딩용 CLIP 모델
- `VLM_MODEL_NAME`: 캡션 생성용 VLM 모델
- `CAPTION_EMBEDDING_MODEL`: 텍스트 검색용 BGE 모델
- `TOP_K_SIMILAR`: 유사한 이미지 상위 K개 (기본값: 5)

---

## 🐛 문제 해결

### 일반적인 오류들

1. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - 더 작은 모델 사용

2. **Hugging Face 토큰 오류**
   - `config.py`에서 `HUGGINGFACE_TOKEN` 주석이 제거되었는지 확인
   - 토큰이 올바른지 확인

3. **파일 누락 오류**
   - 각 단계별로 생성된 파일이 올바른 위치에 있는지 확인
   - `personalized_DB/` 폴더와 `personalized_DB_Embedding/` 폴더 존재 확인

4. **JSON 파싱 오류**
   - `resize_json_files.py`와 `sync_all_captions.py`를 순서대로 실행했는지 확인
   - 백업 파일이 올바른 형식인지 확인

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
