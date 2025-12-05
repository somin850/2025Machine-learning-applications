#!/usr/bin/env python3
"""
Query Maker - OpenAI GPT-4o-mini를 사용하여 이미지 검색 쿼리 생성
이미지와 원본 캡션을 기반으로 인간이 실제 검색할 때 사용할 법한 쿼리를 생성
"""

import json
import os
import base64
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import argparse
from datasets import load_dataset
import config

# OpenAI API 설정
OPENAI_API_KEY = "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"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

class QueryGenerator:
    """OpenAI GPT-4o-mini를 사용한 쿼리 생성 클래스"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        QueryGenerator 초기화
        
        Args:
            api_key (str): OpenAI API 키
        """
        self.api_key = api_key
        self.api_url = OPENAI_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.dataset = None
        
        print(f"🤖 Initializing Query Generator with GPT-4o-mini")
    
    def load_huggingface_dataset(self):
        """Hugging Face 데이터셋 로드"""
        if self.dataset is None:
            print(f"📊 Loading Hugging Face dataset: {config.DATASET_NAME}...")
            try:
                if config.TOTAL_SAMPLES:
                    dataset_split = f"{config.DATASET_SPLIT}[:{config.TOTAL_SAMPLES}]"
                else:
                    dataset_split = config.DATASET_SPLIT
                
                self.dataset = load_dataset(config.DATASET_NAME, split=dataset_split)
                print(f"✅ Dataset loaded successfully. Total samples: {len(self.dataset)}")
                
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                raise
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        이미지를 base64로 인코딩
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def encode_pil_image_to_base64(self, pil_image) -> str:
        """
        PIL Image를 base64로 인코딩 (Hugging Face 데이터셋용)
        
        Args:
            pil_image: PIL Image 객체
            
        Returns:
            str: base64 인코딩된 이미지 문자열
        """
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def create_search_query_prompt(self, caption: str) -> str:
        """
        검색 쿼리 생성을 위한 프롬프트 생성 (이미지 포함)
        
        Args:
            caption (str): VLM 생성 캡션 또는 원본 캡션
            
        Returns:
            str: GPT에게 전달할 프롬프트
        """
        prompt = f"""**System Role:**
You are a "Semantic Search Simulator" testing a high-performance AI photo gallery.
Your goal is to generate **highly specific, descriptive search queries** by directly analyzing the provided image.

**CRITICAL INSTRUCTION:**
You will receive an IMAGE and a CAPTION. You MUST:
1. **Carefully examine the image** to extract visual details that may not be in the caption
2. **Compare the image with the caption** to identify any missing or inaccurate details
3. **Generate a query that captures the most distinctive visual features** visible in the image

**Image Analysis Requirements:**
- **Look at colors carefully:** Note exact shades, patterns, and color combinations
- **Observe spatial relationships:** Positions, orientations, interactions between objects
- **Identify unique details:** Text, logos, specific clothing items, distinctive objects
- **Notice background elements:** Settings, environments, contextual details
- **Capture actions precisely:** Body positions, movements, expressions

**Query Generation Rules:**
1.  **Maximize Specificity:** Include multiple attributes for the main subject.
    * *Instead of:* "man in hat"
    * *Use:* "man in grey t-shirt wearing yellow paper bag hat with text Bite"
2.  **Mandatory Features:** You MUST include:
    * **Specific Colors:** Extract exact colors from the image (e.g., "black race car", "red and white striped shirt")
    * **Unique Actions:** Describe precise actions visible in the image (e.g., "being sprayed with water", "jumping off a dock")
    * **Distinct Objects/Context:** Note text, logos, specific items visible in the image (e.g., "American flag backpack", "words Bite on hat")
    * **Visual Details:** Include details you can see but might not be in the caption
3.  **Phrasing Style:**
    * Do NOT write full grammatical sentences (No "In this image there is...")
    * Write a **Dense Descriptive Phrase** (Noun + Adjectives + Prepositional phrases)
    * Target length: **6 to 12 words** (Keep it concise but distinctive)

**Examples:**

* **Input Caption:** "A man in an orange hat starring at something ."
    * **Output Query:** man in orange hat
    * *(Reasoning: "Orange hat" is the key identifier.)*

* **Input Caption:** "A brown dog is running through neck-deep water carrying a tennis ball ."
    * **Output Query:** brown dog with tennis ball
    * *(Reasoning: "Brown", "water", and "tennis ball" are distinct features.)*

* **Input Caption:** "A boy in a striped shirt is jumping in front of a water fountain ."
    * **Output Query:** boy in striped shirt jumping
    * *(Reasoning: "Striped shirt" distinguishes him from other boys.)*

* **Input Caption:** "A black and white dog catches a toy in midair ."
    * **Output Query:** black and white dog catching toy
    * *(Reasoning: The color pattern "black and white" is crucial.)*

**Reference Caption:**
"{caption}"

**Your Task:**
1. Examine the image carefully
2. Identify the most distinctive visual features that would help find this specific image
3. Generate a search query that combines:
   - Information from the caption (if accurate)
   - Additional visual details you observe in the image
   - Unique characteristics that distinguish this image from similar ones

**Output:**
(Print ONLY the search query phrase, no explanations.):"""

        return prompt
    
    def generate_query_from_image(self, image_path: str, original_caption: str, 
                                 max_retries: int = 3) -> Optional[str]:
        """
        이미지와 캡션으로부터 검색 쿼리 생성
        
        Args:
            image_path (str): 이미지 파일 경로
            original_caption (str): 원본 캡션
            max_retries (int): 최대 재시도 횟수
            
        Returns:
            Optional[str]: 생성된 검색 쿼리 또는 None
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        try:
            # 이미지를 base64로 인코딩
            base64_image = self.encode_image_to_base64(image_path)
            
            # 프롬프트 생성
            prompt = self.create_search_query_prompt(original_caption)
            
            # API 요청 데이터 구성
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            # API 호출 (재시도 로직 포함)
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        OPENAI_API_URL,
                        headers=self.headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        query = result['choices'][0]['message']['content'].strip()
                        
                        # 따옴표 제거 및 정리
                        query = query.strip('"\'')
                        
                        return query
                    else:
                        print(f"⚠️ API Error (attempt {attempt + 1}): {response.status_code}")
                        if attempt == max_retries - 1:
                            print(f"❌ Failed after {max_retries} attempts: {response.text}")
                            return None
                        
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Request Error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return None
                    
        except Exception as e:
            print(f"❌ Error generating query for {image_path}: {e}")
            return None
    
    def generate_query_with_base64_image(self, base64_image: str, original_caption: str, debug: bool = False) -> Optional[str]:
        """
        base64 인코딩된 이미지와 원본 캡션으로부터 검색 쿼리 생성
        
        Args:
            base64_image (str): base64 인코딩된 이미지
            original_caption (str): 원본 캡션
            debug (bool): 디버깅 정보 출력 여부
            
        Returns:
            Optional[str]: 생성된 쿼리 또는 None
        """
        try:
            # 프롬프트 생성
            prompt = self.create_search_query_prompt(original_caption)
            
            # 이미지 크기 확인 (디버깅용)
            if debug:
                image_size_kb = len(base64_image) * 3 / 4 / 1024  # base64는 약 33% 더 큼
                print(f"   📸 Image size: {image_size_kb:.1f} KB (base64 encoded)")
                print(f"   📝 Caption: {original_caption[:60]}...")
                print(f"   💬 Prompt length: {len(prompt)} chars")
            
            # OpenAI API 요청 데이터 구성
            # GPT-4o-mini는 이미지와 텍스트를 함께 받을 수 있음
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            if debug:
                print(f"   📤 Sending request to OpenAI API...")
                print(f"   🔗 Image format: data:image/jpeg;base64,{base64_image[:50]}...")
            
            # API 요청
            response = requests.post(self.api_url, headers=self.headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                query = result['choices'][0]['message']['content'].strip()
                return query
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error generating query: {e}")
            return None
    
    def generate_queries_from_dataset(self, dataset_path: str, image_dir: str, 
                                    output_file: str, max_queries: int = None, resume: bool = False,
                                    vlm_captions_path: str = None) -> Dict:
        """
        데이터셋으로부터 쿼리들을 생성
        
        Args:
            dataset_path (str): 실험 데이터 JSON 파일 경로
            image_dir (str): 이미지 디렉토리 경로
            output_file (str): 출력 쿼리 JSON 파일 경로
            max_queries (int): 생성할 최대 쿼리 수 (None이면 전체)
            resume (bool): 기존 파일에서 이어서 생성할지 여부
            vlm_captions_path (str): VLM 캡션 파일 경로 (None이면 원본 캡션 사용)
            
        Returns:
            Dict: 생성 통계
        """
        print(f"📂 Loading dataset from: {dataset_path}")
        
        # VLM 캡션 로드 (있는 경우)
        vlm_captions = {}
        if vlm_captions_path and os.path.exists(vlm_captions_path):
            print(f"📂 Loading VLM captions from: {vlm_captions_path}")
            with open(vlm_captions_path, 'r', encoding='utf-8') as f:
                vlm_data = json.load(f)
                vlm_captions = vlm_data.get('captions', {})
            print(f"   ✅ Loaded {len(vlm_captions)} VLM captions")
        else:
            print(f"   ℹ️  Using original captions from experiment_data.json")
        
        # 데이터셋 로드
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if isinstance(dataset, list):
            experiments = dataset
        elif 'experiments' in dataset:
            experiments = dataset['experiments']
        else:
            experiments = list(dataset.values())
        
        # 처리할 실험 수 결정
        if max_queries:
            experiments = experiments[:max_queries]
        
        print(f"📊 Processing {len(experiments)} experiments")
        
        # Rate limiting 설정 (1분에 30개 = 2초당 1개)
        RATE_LIMIT_DELAY = 2.0  # 초
        QUERIES_PER_MINUTE = 30
        print(f"⏱️  Rate limiting: {QUERIES_PER_MINUTE} queries/minute ({RATE_LIMIT_DELAY}s delay between queries)")
        
        # 기존 쿼리 로드 (resume 모드인 경우)
        existing_queries = {}
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_queries = existing_data.get('queries', {})
                print(f"📂 Loaded {len(existing_queries)} existing queries from {output_file}")
            except Exception as e:
                print(f"⚠️  Could not load existing queries: {e}")
        
        # 쿼리 생성
        queries = existing_queries.copy()  # 기존 쿼리로 시작
        success_count = len(existing_queries)  # 기존 성공 개수
        failed_count = 0
        skipped_count = 0
        
        # Hugging Face 데이터셋 로드
        self.load_huggingface_dataset()
        
        for i, experiment in enumerate(tqdm(experiments, desc="Generating queries")):
            try:
                # experiment_data.json에서 정보 추출
                experiment_index = experiment.get('experiment_index', i)
                original_index = experiment.get('original_index', i)
                
                # VLM 캡션 사용 (있는 경우), 없으면 원본 캡션 사용
                if vlm_captions and str(experiment_index) in vlm_captions:
                    caption = vlm_captions[str(experiment_index)]
                    caption_source = "VLM"
                else:
                    caption = experiment.get('caption', '')
                    caption_source = "original"
                
                print(f"🔍 Processing experiment {experiment_index}: original_index={original_index} (caption: {caption_source})")
                
                # 이미 생성된 쿼리인지 확인 (resume 모드)
                if str(experiment_index) in queries:
                    skipped_count += 1
                    if skipped_count % 20 == 0:  # 20개마다 출력
                        print(f"⏭️  Skipped {skipped_count} existing queries...")
                    continue
                
                # Hugging Face 데이터셋에서 original_index에 해당하는 이미지 가져오기
                if original_index < len(self.dataset):
                    sample = self.dataset[original_index]
                    pil_image = sample['image']  # PIL Image 객체
                    
                    # PIL Image를 base64로 인코딩
                    base64_image = self.encode_pil_image_to_base64(pil_image)
                    
                    # 쿼리 생성 (이미지 + 캡션 사용)
                    query = self.generate_query_with_base64_image(base64_image, caption)
                else:
                    print(f"⚠️  Original index {original_index} out of range")
                    failed_count += 1
                    continue
                
                if query:
                    # experiment_index를 키로 사용 (0, 1, 2, ...)
                    queries[str(experiment_index)] = query
                    success_count += 1
                    
                    print(f"✅ Generated query for experiment {experiment_index} (original_index={original_index}): {query[:50]}...")
                    
                    # 진행 상황 출력 (5개마다)
                    if (i + 1) % 5 == 0:
                        print(f"   Progress: {i + 1}/{len(experiments)} - Success: {success_count}, Failed: {failed_count}")
                else:
                    failed_count += 1
                    print(f"   ❌ Failed to generate query for experiment {experiment_index}")
                
                # Rate limiting: 1분에 30개 = 2초당 1개
                # (건너뛴 쿼리는 딜레이 없음)
                if str(experiment_index) not in existing_queries:
                    time.sleep(RATE_LIMIT_DELAY)  # Rate limit 대기
                
            except Exception as e:
                failed_count += 1
                print(f"   ❌ Error processing experiment {i}: {e}")
                # 에러 발생 시에도 rate limit 준수를 위해 대기
                time.sleep(RATE_LIMIT_DELAY)
                continue
        
        # 결과 저장
        query_data = {
            "queries": queries,
            "metadata": {
                "total_experiments": len(experiments),
                "successful_queries": success_count,
                "failed_queries": failed_count,
                "skipped_queries": skipped_count,
                "newly_generated": success_count - len(existing_queries),
                "success_rate": success_count / len(experiments) if experiments else 0,
                "generation_time": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "dataset_source": dataset_path,
                "image_directory": image_dir,
                "vlm_captions_source": vlm_captions_path if vlm_captions_path else "original_captions",
                "caption_type": "VLM" if vlm_captions else "original",
                "resume_mode": resume
            }
        }
        
        # 출력 디렉토리 생성 (디렉토리가 있는 경우에만)
        output_dir = os.path.dirname(output_file)
        if output_dir:  # 디렉토리 경로가 있는 경우에만 생성
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(query_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Queries saved to: {output_file}")
        
        # 통계 반환
        success_rate = success_count / len(experiments) if experiments else 0
        newly_generated = success_count - len(existing_queries)
        stats = {
            "total_experiments": len(experiments),
            "successful_queries": success_count,
            "failed_queries": failed_count,
            "skipped_queries": skipped_count,
            "newly_generated": newly_generated,
            "success_rate": success_rate,
            "output_file": output_file
        }
        
        print(f"\n📊 Generation Statistics:")
        print(f"   Total experiments: {stats['total_experiments']}")
        print(f"   Successful queries: {stats['successful_queries']}")
        print(f"   Failed queries: {stats['failed_queries']}")
        print(f"   Skipped (existing): {stats['skipped_queries']}")
        print(f"   Newly generated: {newly_generated}")
        print(f"   Success rate: {success_rate:.2%}")
        
        return stats

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Generate search queries using GPT-4o-mini")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to experiment dataset JSON file")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output", type=str, default="generated_queries.json",
                       help="Output query JSON file path")
    parser.add_argument("--max_queries", type=int, default=None,
                       help="Maximum number of queries to generate (default: all)")
    parser.add_argument("--api_key", type=str, default=OPENAI_API_KEY,
                       help="OpenAI API key")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🤖 Search Query Generation with GPT-4o-mini")
    print("=" * 80)
    print(f"📂 Dataset: {args.dataset}")
    print(f"🖼️  Image directory: {args.image_dir}")
    print(f"💾 Output file: {args.output}")
    print(f"🔢 Max queries: {args.max_queries or 'All'}")
    print()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory not found: {args.image_dir}")
        return
    
    try:
        # 쿼리 생성기 초기화
        generator = QueryGenerator(api_key=args.api_key)
        
        # 쿼리 생성 실행
        stats = generator.generate_queries_from_dataset(
            dataset_path=args.dataset,
            image_dir=args.image_dir,
            output_file=args.output,
            max_queries=args.max_queries
        )
        
        if stats['successful_queries'] > 0:
            print(f"\n🎉 Query generation completed!")
            print(f"   Generated {stats['successful_queries']} queries")
            print(f"   Success rate: {stats['success_rate']:.2%}")
        else:
            print(f"\n❌ No queries were generated successfully!")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
