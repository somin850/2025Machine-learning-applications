# 📄 prompt_generator.py
"""
VLM용 프롬프트 생성 모듈
유사한 이미지들의 캡션을 기반으로 VLM용 프롬프트를 생성합니다.
"""

from typing import List, Dict


class PromptGenerator:
    """VLM용 프롬프트를 생성하는 클래스"""
    
    def __init__(self, template: str = None):
        """
        PromptGenerator 초기화
        
        Args:
            template (str): 프롬프트 템플릿 (기본값: 내장 템플릿 사용)
        """
        if template is None:
            self.template = self._get_default_template()
        else:
            self.template = template
    
    def _get_default_template(self) -> str:
        """기본 프롬프트 템플릿을 반환합니다."""
        return """The similar Image have these captions

{similar_captions}

Based on these similar image captions above, please generate an accurate and detailed caption for the input image. The caption should be in English"""
    
    def generate_prompt(self, similar_captions: List[str], debug: bool = False) -> str:
        """
        유사한 캡션들을 기반으로 VLM용 프롬프트를 생성합니다.
        
        Args:
            similar_captions (List[str]): 유사한 이미지들의 캡션 리스트
            debug (bool): 디버깅 출력 여부
        
        Returns:
            str: 생성된 프롬프트
        """
        if not similar_captions:
            return "Based on these similar image captions above, please generate an accurate and detailed caption for the input image. The caption should be in English"
        
        # 캡션들을 번호와 함께 포맷팅
        formatted_captions = []
        for i, caption in enumerate(similar_captions, 1):
            formatted_captions.append(f"{i}. {caption}")
        
        captions_text = "\n".join(formatted_captions)
        
        # 디버깅 출력
        if debug:
            print("\n" + "=" * 60)
            print("📝 Generated Prompt Structure:")
            print("=" * 60)
            print(f"\n[Input Captions ({len(similar_captions)} items)]:")
            for i, caption in enumerate(similar_captions, 1):
                print(f"  {i}. {caption}")
            print(f"\n[Formatted Captions Text]:")
            print(captions_text)
            print(f"\n[Full Prompt]:")
        
        # 템플릿에 캡션들 삽입
        prompt = self.template.format(similar_captions=captions_text)
        
        if debug:
            print(prompt)
            print("=" * 60 + "\n")
        
        return prompt
    
    def generate_prompt_from_search_result(self, search_result: Dict, db_manager=None, 
                                          similarity_threshold: float = 0.75, max_captions: int = 3) -> str:
        """
        검색 결과로부터 동적 프롬프트를 생성합니다.
        유사도 임계값 이상의 이미지가 있으면 해당 캡션들을 사용하고,
        없으면 기본 프롬프트를 반환합니다.
        
        Args:
            search_result (Dict): similarity_search에서 반환된 검색 결과
            db_manager: DatabaseManager 인스턴스 (캡션 검색용)
            similarity_threshold (float): 유사도 임계값 (기본값: 0.75)
            max_captions (int): 최대 캡션 개수 (기본값: 3)
        
        Returns:
            str: 생성된 프롬프트
        """
        similar_images = search_result.get('similar_images', [])
        
        # 유사도 임계값 이상의 이미지들만 필터링
        high_similarity_images = [
            img for img in similar_images 
            if img.get('similarity', 0.0) >= similarity_threshold
        ]
        
        # 유사도 임계값 이상의 이미지가 없으면 기본 프롬프트 반환
        if not high_similarity_images:
            return "Based on these similar image captions above, please generate an accurate and detailed caption for the input image. The caption should be in English"
        
        # 최대 개수만큼 선택 (이미 유사도 순으로 정렬되어 있음)
        selected_images = high_similarity_images[:max_captions]
        
        # 유사한 이미지들의 캡션 추출
        if db_manager:
            # DB 매니저를 사용하여 캡션 검색 (원본 + 생성된 캡션)
            indices = [img_info['index'] for img_info in selected_images]
            captions = db_manager.get_captions_by_indices(indices)
        else:
            # 기존 방식: 메타데이터에서 캡션 추출
            captions = []
            for img_info in selected_images:
                metadata = img_info.get('metadata', {})
                caption = metadata.get('caption', 'No caption available')
                captions.append(caption)
        
        return self.generate_prompt(captions)
    
    def set_template(self, template: str):
        """
        프롬프트 템플릿을 설정합니다.
        
        Args:
            template (str): 새로운 프롬프트 템플릿
        """
        self.template = template
    
    def get_template(self) -> str:
        """현재 프롬프트 템플릿을 반환합니다."""
        return self.template


class AdvancedPromptGenerator(PromptGenerator):
    """고급 프롬프트 생성 기능을 제공하는 클래스"""
    
    def __init__(self, template: str = None, include_similarity_scores: bool = False):
        """
        AdvancedPromptGenerator 초기화
        
        Args:
            template (str): 프롬프트 템플릿
            include_similarity_scores (bool): 유사도 점수 포함 여부
        """
        super().__init__(template)
        self.include_similarity_scores = include_similarity_scores
    
    def _get_default_template(self) -> str:
        """고급 기본 프롬프트 템플릿을 반환합니다."""
        return """다음은 입력 이미지와 유사한 이미지들의 캡션입니다:

{similar_captions}

위의 유사한 이미지들의 캡션을 참고하여, 입력된 이미지에 대한 정확하고 상세한 캡션을 생성해주세요.

요구사항:
1. 캡션은 영어로 작성해주세요
2. 이미지의 주요 객체, 행동, 배경을 포함해주세요
3. 색상, 위치, 감정 등의 세부사항도 포함해주세요
4. 간결하면서도 정보가 풍부한 캡션을 만들어주세요
5. 유사한 캡션들의 패턴을 참고하되, 입력 이미지만의 고유한 특징도 반영해주세요"""
    
    def generate_prompt_with_scores(self, similar_images_info: List[Dict]) -> str:
        """
        유사도 점수를 포함한 프롬프트를 생성합니다.
        
        Args:
            similar_images_info (List[Dict]): 유사한 이미지들의 정보 (유사도 점수 포함)
        
        Returns:
            str: 생성된 프롬프트
        """
        if not similar_images_info:
            return "Please generate a detailed caption for this image in English."
        
        # 캡션들을 유사도 점수와 함께 포맷팅
        formatted_captions = []
        for i, img_info in enumerate(similar_images_info, 1):
            metadata = img_info.get('metadata', {})
            caption = metadata.get('caption', 'No caption available')
            similarity = img_info.get('similarity', 0.0)
            
            if self.include_similarity_scores:
                formatted_captions.append(f"{i}. (유사도: {similarity:.3f}) {caption}")
            else:
                formatted_captions.append(f"{i}. {caption}")
        
        captions_text = "\n".join(formatted_captions)
        
        # 템플릿에 캡션들 삽입
        prompt = self.template.format(similar_captions=captions_text)
        
        return prompt
    
    def generate_contextual_prompt(self, search_result: Dict, context: str = None) -> str:
        """
        컨텍스트를 포함한 프롬프트를 생성합니다.
        
        Args:
            search_result (Dict): 검색 결과
            context (str): 추가 컨텍스트 정보
        
        Returns:
            str: 생성된 프롬프트
        """
        base_prompt = self.generate_prompt_with_scores(search_result.get('similar_images', []))
        
        if context:
            contextual_prompt = f"{base_prompt}\n\n추가 컨텍스트: {context}"
            return contextual_prompt
        
        return base_prompt


def create_prompt_generator(template: str = None):
    """PromptGenerator 인스턴스를 생성하는 편의 함수"""
    return PromptGenerator(template)


def create_advanced_prompt_generator(template: str = None, include_similarity_scores: bool = False):
    """AdvancedPromptGenerator 인스턴스를 생성하는 편의 함수"""
    return AdvancedPromptGenerator(template, include_similarity_scores)


def generate_vlm_prompt(search_result: Dict, template: str = None, 
                       include_scores: bool = False, db_manager=None,
                       similarity_threshold: float = 0.75, max_captions: int = 3) -> str:
    """
    검색 결과로부터 VLM용 동적 프롬프트를 생성하는 편의 함수
    
    Args:
        search_result (Dict): 검색 결과
        template (str): 프롬프트 템플릿
        include_scores (bool): 유사도 점수 포함 여부
        db_manager: DatabaseManager 인스턴스
        similarity_threshold (float): 유사도 임계값 (기본값: 0.75)
        max_captions (int): 최대 캡션 개수 (기본값: 3)
    
    Returns:
        str: 생성된 프롬프트
    """
    if include_scores:
        generator = create_advanced_prompt_generator(template, include_scores)
        return generator.generate_prompt_with_scores(search_result.get('similar_images', []))
    else:
        generator = create_prompt_generator(template)
        return generator.generate_prompt_from_search_result(
            search_result, db_manager, similarity_threshold, max_captions
        )
