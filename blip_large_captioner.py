# 📄 blip_large_captioner.py
"""
BLIP Large 모델을 사용하여 400개 실험 이미지의 캡션을 생성합니다.
결과는 blip_large_captions.json 파일에 저장됩니다.
"""

import os
import json
from datetime import datetime
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import config
from dataset_loader import Flickr8KLoader

class BlipLargeCaptioner:
    """BLIP Large 모델을 사용한 캡션 생성기"""
    
    def __init__(self, device=None):
        """
        BlipLargeCaptioner 초기화
        
        Args:
            device: 사용할 장치 (cuda 또는 cpu)
        """
        if device is None:
            device = config.get_device()
        
        self.device = device
        self.model_name = "Salesforce/blip-image-captioning-large"
        
        print(f"Loading BLIP Large model: {self.model_name} on {device}...")
        
        # 모델과 프로세서 로드
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        
        # Large 모델은 메모리를 많이 사용하므로 최적화 설정
        if device.type == 'cuda':
            print("  - Enabling GPU optimizations for Large model")
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # FP16으로 메모리 절약
                low_cpu_mem_usage=True
            )
        else:
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        
        self.model.to(device)
        self.model.eval()
        
        print("BLIP Large model loaded successfully.")
    
    def generate_caption(self, image, max_length=70, num_beams=5):
        """
        단일 이미지에 대한 캡션을 생성합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            max_length: 최대 캡션 길이 (Large 모델은 더 긴 캡션 생성 가능)
            num_beams: 빔 서치 크기 (Large 모델은 더 많은 빔 사용)
            
        Returns:
            str: 생성된 캡션
        """
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be PIL Image or image path")
        
        # 입력 전처리
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # GPU에서 FP16 사용 시 입력도 변환
        if self.device.type == 'cuda':
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].half()
        
        # 캡션 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                length_penalty=1.0,  # Large 모델에서 더 자세한 캡션 생성
                repetition_penalty=1.2
            )
        
        # 텍스트 디코딩
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption
    
    def generate_captions_for_experiments(self, experiment_data, output_file="blip_large_captions.json"):
        """
        실험 데이터의 모든 이미지에 대한 캡션을 생성합니다.
        
        Args:
            experiment_data: 실험 데이터 리스트
            output_file: 출력 파일 경로
            
        Returns:
            dict: 생성된 캡션 데이터
        """
        print(f"\n🎯 Generating captions for {len(experiment_data)} experiment images...")
        print(f"Model: {self.model_name}")
        print(f"Output file: {output_file}")
        print("=" * 60)
        
        captions_data = {
            "captions": {},
            "metadata": {},
            "total_captions": 0,
            "model_info": {
                "model_name": self.model_name,
                "generation_timestamp": datetime.now().isoformat(),
                "total_experiments": len(experiment_data),
                "model_size": "large",
                "optimization": "fp16" if self.device.type == 'cuda' else "fp32"
            }
        }
        
        for i, item in enumerate(experiment_data):
            try:
                print(f"\n📸 Processing experiment {i}/{len(experiment_data)-1} (Index: {i})")
                
                # 이미지 로드
                image = item['image']
                original_caption = item['caption']
                original_index = item['original_index']
                
                # 캡션 생성 (Large 모델은 더 긴 캡션과 더 많은 빔 사용)
                generated_caption = self.generate_caption(image, max_length=70, num_beams=5)
                
                print(f"  Original: {original_caption}")
                print(f"  Generated: {generated_caption}")
                
                # 데이터 저장
                experiment_index = str(i)  # 실험 인덱스를 키로 사용
                
                captions_data["captions"][experiment_index] = generated_caption
                captions_data["metadata"][experiment_index] = {
                    "experiment_index": i,
                    "original_index": original_index,
                    "original_caption": original_caption,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "model_size": "large"
                }
                
                # GPU 메모리 정리 (Large 모델은 더 자주 정리)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # 진행률 표시
                if (i + 1) % 50 == 0:
                    print(f"  📊 Progress: {i+1}/{len(experiment_data)} ({(i+1)/len(experiment_data)*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error processing experiment {i}: {e}")
                # Large 모델에서 메모리 부족 시 정리
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        # 총 캡션 수 업데이트
        captions_data["total_captions"] = len(captions_data["captions"])
        
        # 파일 저장
        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully generated {captions_data['total_captions']} captions")
        print(f"📁 Results saved to: {output_file}")
        
        return captions_data


def main():
    """메인 함수"""
    try:
        # GPU 메모리 설정
        config.setup_gpu_memory()
        
        print("🚀 BLIP Large Caption Generation")
        print("=" * 60)
        
        # 데이터셋 로더 초기화
        print("\n--- Step 1: Loading Dataset ---")
        dataset_loader = Flickr8KLoader()
        dataset_loader.load_dataset()
        dataset_loader.load_split_data()
        
        # 실험 데이터 가져오기
        experiment_data = dataset_loader.get_experiment_data()
        print(f"✓ Loaded {len(experiment_data)} experiment images")
        
        # BLIP Large 캡셔너 초기화
        print("\n--- Step 2: Initializing BLIP Large Model ---")
        captioner = BlipLargeCaptioner()
        
        # 캡션 생성
        print("\n--- Step 3: Generating Captions ---")
        results = captioner.generate_captions_for_experiments(
            experiment_data, 
            output_file="blip_large_captions.json"
        )
        
        print(f"\n🎉 BLIP Large caption generation completed!")
        print(f"📊 Statistics:")
        print(f"  - Total experiments: {len(experiment_data)}")
        print(f"  - Successful captions: {results['total_captions']}")
        print(f"  - Success rate: {results['total_captions']/len(experiment_data)*100:.1f}%")
        
        # 최종 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
