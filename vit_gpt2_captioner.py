# 📄 vit_gpt2_captioner.py
"""
ViT-GPT2 모델을 사용하여 400개 실험 이미지의 캡션을 생성합니다.
결과는 vit_gpt2_captions.json 파일에 저장됩니다.
"""

import os
import json
from datetime import datetime
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import config
from dataset_loader import Flickr8KLoader

# OpenMP 오류 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ViTGPT2Captioner:
    """ViT-GPT2 모델을 사용한 캡션 생성기"""
    
    def __init__(self, device=None):
        """
        ViTGPT2Captioner 초기화
        
        Args:
            device: 사용할 장치 (cuda 또는 cpu)
        """
        if device is None:
            device = config.get_device()
        
        self.device = device
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        # Mixed precision 설정 저장
        self.use_mixed_precision = getattr(config, 'USE_MIXED_PRECISION', True)
        
        print(f"Loading ViT-GPT2 model: {self.model_name} on {device}...")
        
        try:
            # 모델, 프로세서, 토크나이저 로드
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # GPU 최적화 설정
            if device.type == 'cuda':
                print("  - Enabling GPU optimizations for ViT-GPT2 model")
                self.model = self.model.to(device)
                if self.use_mixed_precision:
                    self.model = self.model.half()  # FP16으로 메모리 절약
                    print("  - Mixed precision (FP16) enabled")
            else:
                self.model = self.model.to(device)
            
            self.model.eval()
            
            # 생성 설정
            self.max_length = 16
            self.num_beams = 4
            
            print("ViT-GPT2 model loaded successfully.")
            
        except Exception as e:
            print(f"❌ Failed to load ViT-GPT2 model: {e}")
            print("💡 Possible solutions:")
            print("  1. Check internet connection")
            print("  2. Install required packages: pip install transformers torch pillow")
            raise
    
    def generate_caption(self, image, max_length=None, num_beams=None):
        """
        단일 이미지에 대한 캡션을 생성합니다.
        
        Args:
            image: PIL Image 객체 또는 이미지 경로
            max_length: 최대 캡션 길이
            num_beams: 빔 서치 크기
            
        Returns:
            str: 생성된 캡션
        """
        # 기본값 설정
        if max_length is None:
            max_length = self.max_length
        if num_beams is None:
            num_beams = self.num_beams
        
        # 이미지 로드
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be PIL Image or image path")
        
        # 이미지 전처리
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # GPU에서 FP16 사용 시 입력도 변환
        if self.device.type == 'cuda' and self.use_mixed_precision:
            pixel_values = pixel_values.half()
        
        # 캡션 생성
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                do_sample=False,
                early_stopping=True
            ).sequences
        
        # 텍스트 디코딩
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return caption.strip()
    
    def generate_captions_for_experiments(self, experiment_data, output_file="vit_gpt2_captions.json"):
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
        
        # 기존 파일이 있으면 로드하여 이어서 작업
        captions_data = {
            "captions": {},
            "metadata": {},
            "total_captions": 0,
            "model_info": {
                "model_name": self.model_name,
                "generation_timestamp": datetime.now().isoformat(),
                "total_experiments": len(experiment_data),
                "model_type": "vit-gpt2",
                "max_length": self.max_length,
                "num_beams": self.num_beams
            }
        }
        
        # 기존 파일 로드 (있는 경우)
        if os.path.exists(output_file):
            print(f"Loading existing captions from {output_file}...")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    captions_data['captions'] = {str(k): v for k, v in existing_data.get('captions', {}).items()}
                    captions_data['metadata'] = {str(k): v for k, v in existing_data.get('metadata', {}).items()}
                    captions_data['total_captions'] = existing_data.get('total_captions', 0)
                print(f"✓ Loaded {captions_data['total_captions']} existing captions.")
            except Exception as e:
                print(f"⚠ Error loading existing captions: {e}. Starting fresh.")
                captions_data['captions'] = {}
                captions_data['metadata'] = {}
                captions_data['total_captions'] = 0
        
        successful_captions = 0
        failed_captions = 0
        
        for i, item in enumerate(experiment_data):
            try:
                # 이미 캡션이 생성된 경우 건너뛰기
                if str(i) in captions_data['captions']:
                    print(f"  Skipping experiment {i}: caption already exists.")
                    continue
                
                print(f"\n📸 Processing experiment {i}/{len(experiment_data)-1} (Index: {i})")
                
                # 이미지 로드
                image = item['image']
                original_caption = item['caption']
                original_index = item['original_index']
                
                # 캡션 생성
                generated_caption = self.generate_caption(image)
                
                print(f"  Original: {original_caption}")
                print(f"  Generated: {generated_caption}")
                
                # 데이터 저장
                captions_data["captions"][str(i)] = generated_caption
                captions_data["metadata"][str(i)] = {
                    "experiment_index": i,
                    "original_index": original_index,
                    "original_caption": original_caption,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "model_type": "vit-gpt2"
                }
                
                successful_captions += 1
                
                # 중간 저장 (50개마다)
                if (i + 1) % 50 == 0:
                    captions_data["total_captions"] = len(captions_data["captions"])
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(captions_data, f, ensure_ascii=False, indent=2)
                    print(f"  📊 Progress: {i+1}/{len(experiment_data)} ({(i+1)/len(experiment_data)*100:.1f}%)")
                    print(f"  💾 Intermediate save completed")
                
                # GPU 메모리 정리
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing experiment {i}: {e}")
                failed_captions += 1
                
                # GPU 메모리 정리
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        # 총 캡션 수 업데이트
        captions_data["total_captions"] = len(captions_data["captions"])
        
        # 최종 파일 저장
        print(f"\n💾 Saving final results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Caption generation completed!")
        print(f"📊 Results:")
        print(f"  - Total experiments: {len(experiment_data)}")
        print(f"  - Successful captions: {successful_captions}")
        print(f"  - Failed captions: {failed_captions}")
        print(f"  - Total captions in file: {captions_data['total_captions']}")
        print(f"📁 Results saved to: {output_file}")
        
        return captions_data


def main():
    """메인 함수"""
    try:
        print("🚀 ViT-GPT2 Caption Generation")
        print("=" * 60)
        
        # 데이터셋 로더 초기화
        print("\n--- Step 1: Loading Dataset ---")
        dataset_loader = Flickr8KLoader()
        
        # 데이터셋 로드
        print("  Loading Flickr8K dataset...")
        dataset_loader.load_dataset()
        print(f"  ✓ Dataset loaded: {len(dataset_loader.dataset) if dataset_loader.dataset else 0} total samples")
        
        # 데이터 분할 로드
        print("  Loading split data...")
        dataset_loader.load_split_data()
        
        # 실험 데이터 가져오기
        experiment_data = dataset_loader.get_experiment_data()
        print(f"✓ Loaded {len(experiment_data)} experiment images")
        
        # 실험 데이터가 비어있는지 확인
        if len(experiment_data) == 0:
            print("❌ No experiment data found!")
            print("💡 Possible solutions:")
            print("  1. Check if data/experiment_data.json exists")
            print("  2. Run main.py first to initialize the system")
            print("  3. Check dataset loading configuration")
            return
        
        # ViT-GPT2 캡셔너 초기화
        print("\n--- Step 2: Initializing ViT-GPT2 Model ---")
        captioner = ViTGPT2Captioner()
        
        # 캡션 생성
        print("\n--- Step 3: Generating Captions ---")
        results = captioner.generate_captions_for_experiments(
            experiment_data, 
            output_file="vit_gpt2_captions.json"
        )
        
        print(f"\n🎉 ViT-GPT2 caption generation completed!")
        print(f"📊 Statistics:")
        print(f"  - Total experiments: {len(experiment_data)}")
        print(f"  - Successful captions: {results['total_captions']}")
        
        # ZeroDivisionError 방지
        if len(experiment_data) > 0:
            success_rate = results['total_captions']/len(experiment_data)*100
            print(f"  - Success rate: {success_rate:.1f}%")
        else:
            print(f"  - Success rate: N/A (no experiments)")
        
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
