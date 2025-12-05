# 📄 vlm_captioner.py
"""
SmolVLM 기반 캡션 생성 모듈
프롬프트와 이미지를 입력받아 SmolVLM으로 캡션을 생성합니다.
SmolVLM은 2B 파라미터의 효율적인 Vision Language Model입니다.
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import config


class VLMCaptioner:
    """SmolVLM을 사용하여 캡션을 생성하는 클래스"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        VLMCaptioner 초기화
        
        Args:
            model_name (str): 사용할 SmolVLM 모델 이름
            device (str): 사용할 장치
        """
        if model_name is None:
            model_name = config.VLM_MODEL_NAME
        
        if device is None:
            device = config.get_device()
        
        self.device = device
        self.model_name = model_name
        
        # 모델 접근 권한 확인 및 토큰 설정
        config.check_model_access(model_name)
        if not config.setup_huggingface_token():
            print("⚠ Proceeding without token - some models may fail to load.")
        
        print(f"Loading SmolVLM model: {model_name} on {device}...")
        
        try:
            # SmolVLM 모델 로드
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # A100 GPU 최적화 설정
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
            }
            
            # 기본 attention 사용 (안정성 우선)
            model_kwargs["_attn_implementation"] = "eager"
            
            if device.type == "cuda":
                print("  - Using eager attention for stable performance")
                
                # A100 MIG에서 메모리 최적화
                model_kwargs["low_cpu_mem_usage"] = True
                model_kwargs["device_map"] = "auto"
                print("  - Memory optimization enabled for A100 MIG")
            
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
            
            if device.type != "cuda":  # device_map="auto"가 아닌 경우에만 수동 이동
                self.model = self.model.to(device)
            
            self.model.eval()
            print("SmolVLM model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Possible solutions:")
            print("  1. Set your Hugging Face token in config.py")
            print("  2. Accept the model license at: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct")
            print("  3. Use alternative model: Salesforce/blip-image-captioning-large")
            raise
    
    def generate_caption(self, image, prompt: str = None, 
                        max_new_tokens: int = None, temperature: float = None) -> str:
        """
        이미지와 프롬프트로부터 캡션을 생성합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            prompt (str): 텍스트 프롬프트 (선택사항)
            max_new_tokens (int): 최대 새 토큰 수
            temperature (float): 생성 온도
        
        Returns:
            str: 생성된 캡션
        """
        # 기본값 설정
        if max_new_tokens is None:
            max_new_tokens = config.MAX_LENGTH
        if temperature is None:
            temperature = config.TEMPERATURE
        
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be PIL Image or image path")
        
        # SmolVLM용 메시지 형식 구성
        if prompt:
            # 프롬프트가 있는 경우
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            # 기본 캡션 생성 프롬프트
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in detail. Provide a clear, concise caption in English that describes the main objects, people, actions, and setting visible in the image."}
                    ]
                }
            ]
        
        # 채팅 템플릿 적용
        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # 입력 전처리
        inputs = self.processor(text=formatted_prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # 캡션 생성
        with torch.no_grad():
            if temperature > 0:
                # 온도 기반 샘플링
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )
            else:
                # 그리디 디코딩
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
        
        # 디코딩
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # 응답에서 캡션 부분만 추출
        full_response = generated_texts[0]
        # SmolVLM의 응답에서 실제 캡션 부분만 추출
        if "Assistant:" in full_response:
            caption = full_response.split("Assistant:")[-1].strip()
        else:
            caption = full_response.strip()
        
        return caption
    
    def generate_caption_with_context(self, image, similar_captions: list, 
                                    max_new_tokens: int = None, temperature: float = None) -> str:
        """
        유사한 캡션들을 컨텍스트로 사용하여 캡션을 생성합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            similar_captions (list): 유사한 이미지들의 캡션 리스트
            max_new_tokens (int): 최대 새 토큰 수
            temperature (float): 생성 온도
        
        Returns:
            str: 생성된 캡션
        """
        # 프롬프트 생성
        from prompt_generator import create_prompt_generator
        prompt_generator = create_prompt_generator()
        prompt = prompt_generator.generate_prompt(similar_captions, debug=True)
        
        # 캡션 생성
        caption = self.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return caption
    
    def generate_caption_from_search_result(self, image, search_result: dict, 
                                          max_new_tokens: int = None, temperature: float = None,
                                          db_manager=None, similarity_threshold: float = 0.75) -> str:
        """
        검색 결과를 사용하여 동적 프롬프트로 캡션을 생성합니다.
        유사도 임계값 이상의 이미지가 있으면 Top 3 캡션을 사용하고,
        없으면 기본 프롬프트를 사용합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            search_result (dict): similarity_search에서 반환된 검색 결과
            max_new_tokens (int): 최대 새 토큰 수
            temperature (float): 생성 온도
            db_manager: DatabaseManager 인스턴스 (캡션 검색용)
            similarity_threshold (float): 유사도 임계값 (기본값: 0.75)
        
        Returns:
            str: 생성된 캡션
        """
        # 동적 프롬프트 생성
        from prompt_generator import generate_vlm_prompt
        
        prompt = generate_vlm_prompt(
            search_result=search_result,
            db_manager=db_manager,
            similarity_threshold=similarity_threshold,
            max_captions=3  # Top 3 캡션만 사용
        )
        
        print(f"\n📝 Generated Prompt Strategy:")
        similar_images = search_result.get('similar_images', [])
        high_similarity_count = sum(1 for img in similar_images if img.get('similarity', 0.0) >= similarity_threshold)
        
        if high_similarity_count > 0:
            print(f"  - Using {min(high_similarity_count, 3)} high-similarity captions (threshold: {similarity_threshold})")
            print(f"  - Prompt includes similar image examples")
        else:
            print(f"  - No images above threshold ({similarity_threshold})")
            print(f"  - Using default prompt without examples")
        
        # 캡션 생성
        return self.generate_caption(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )


class AdvancedVLMCaptioner(VLMCaptioner):
    """고급 SmolVLM 캡션 생성 기능을 제공하는 클래스"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """AdvancedVLMCaptioner 초기화"""
        super().__init__(model_name, device)
    
    def generate_multiple_captions(self, image, prompt: str = None, 
                                 num_candidates: int = 3,
                                 max_new_tokens: int = None, temperature: float = 0.7) -> list:
        """
        여러 개의 캡션을 생성합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            prompt (str): 텍스트 프롬프트
            num_candidates (int): 생성할 캡션 개수
            max_new_tokens (int): 최대 새 토큰 수
            temperature (float): 생성 온도
        
        Returns:
            list: 생성된 캡션들의 리스트
        """
        captions = []
        
        # 여러 번 생성하여 다양한 캡션 획득
        for i in range(num_candidates):
            # 각 생성마다 약간 다른 온도 사용
            current_temp = temperature + (i * 0.1)
            caption = self.generate_caption(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=current_temp
            )
            captions.append(caption)
        
        return captions
    
    def generate_best_caption(self, image, similar_captions: list,
                            num_candidates: int = 3,
                            max_new_tokens: int = None, temperature: float = None) -> dict:
        """
        여러 후보 중에서 최고의 캡션을 선택합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            similar_captions (list): 유사한 이미지들의 캡션 리스트
            num_candidates (int): 후보 캡션 개수
            max_new_tokens (int): 최대 새 토큰 수
            temperature (float): 생성 온도
        
        Returns:
            dict: {'best_caption': str, 'all_candidates': list}
        """
        # 프롬프트 생성
        from prompt_generator import create_prompt_generator
        prompt_generator = create_prompt_generator()
        prompt = prompt_generator.generate_prompt(similar_captions)
        
        # 여러 캡션 생성
        candidates = self.generate_multiple_captions(
            image=image,
            prompt=prompt,
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature else 0.7
        )
        
        # 첫 번째 캡션을 최고로 선택 (가장 일관된 결과)
        best_caption = candidates[0] if candidates else ""
        
        return {
            'best_caption': best_caption,
            'all_candidates': candidates
        }


def create_vlm_captioner(model_name: str = None, device: str = None):
    """VLMCaptioner 인스턴스를 생성하는 편의 함수"""
    return VLMCaptioner(model_name, device)


def create_advanced_vlm_captioner(model_name: str = None, device: str = None):
    """AdvancedVLMCaptioner 인스턴스를 생성하는 편의 함수"""
    return AdvancedVLMCaptioner(model_name, device)


def generate_caption_with_similarity(image, search_result: dict, 
                                   model_name: str = None, device: str = None,
                                   max_new_tokens: int = None, temperature: float = None,
                                   db_manager=None, similarity_threshold: float = 0.75) -> str:
    """
    유사도 검색 결과를 사용하여 동적 프롬프트로 캡션을 생성하는 편의 함수
    
    Args:
        image: PIL Image 객체 또는 이미지 경로
        search_result (dict): 검색 결과
        model_name (str): SmolVLM 모델 이름
        device (str): 사용할 장치
        max_new_tokens (int): 최대 새 토큰 수
        temperature (float): 생성 온도
        db_manager: DatabaseManager 인스턴스 (캡션 검색용)
        similarity_threshold (float): 유사도 임계값 (기본값: 0.75)
    
    Returns:
        str: 생성된 캡션
    """
    captioner = create_vlm_captioner(model_name, device)
    return captioner.generate_caption_from_search_result(
        image=image,
        search_result=search_result,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        db_manager=db_manager,
        similarity_threshold=similarity_threshold
    )
