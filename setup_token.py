# 📄 setup_token.py
"""
Hugging Face 토큰 설정 도우미 스크립트
"""

import os
import sys

def setup_token_interactive():
    """대화형으로 Hugging Face 토큰을 설정합니다."""
    print("🔐 Hugging Face Token Setup")
    print("=" * 40)
    
    print("\n📋 토큰이 필요한 이유:")
    print("  - EmbeddingGemma-300M: 제한된 접근 모델")
    print("  - SmolVLM-Instruct: 제한된 접근 모델")
    
    print("\n🔗 토큰 발급 방법:")
    print("  1. https://huggingface.co/settings/tokens 방문")
    print("  2. 'New token' 클릭")
    print("  3. 'Read' 권한으로 토큰 생성")
    print("  4. 생성된 토큰 복사")
    
    print("\n📝 모델 라이선스 동의:")
    print("  - https://huggingface.co/google/embeddinggemma-300m")
    print("  - https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct")
    print("  각 모델 페이지에서 라이선스에 동의해주세요.")
    
    print("\n" + "=" * 40)
    
    # 토큰 입력 받기
    token = input("Hugging Face 토큰을 입력하세요 (또는 Enter로 건너뛰기): ").strip()
    
    if not token:
        print("⚠ 토큰 없이 진행합니다. 일부 모델이 로드되지 않을 수 있습니다.")
        return False
    
    # 환경변수로 설정
    os.environ['HUGGINGFACE_TOKEN'] = token
    
    # config.py 파일 업데이트
    try:
        config_path = "config.py"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 토큰 라인 찾아서 업데이트
            lines = content.split('\n')
            updated = False
            
            for i, line in enumerate(lines):
                if 'HUGGINGFACE_TOKEN = ' in line and 'os.getenv' not in line:
                    lines[i] = f'# HUGGINGFACE_TOKEN = "{token}"  # 보안상 환경변수 사용 권장'
                    updated = True
                    break
            
            if updated:
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print("✓ config.py 파일이 업데이트되었습니다.")
    
    except Exception as e:
        print(f"⚠ config.py 업데이트 실패: {e}")
    
    # 토큰 테스트
    try:
        from huggingface_hub import login
        login(token=token)
        print("✅ 토큰이 성공적으로 설정되었습니다!")
        return True
    except ImportError:
        print("📦 huggingface_hub 설치 중...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import login
        login(token=token)
        print("✅ 토큰이 성공적으로 설정되었습니다!")
        return True
    except Exception as e:
        print(f"❌ 토큰 설정 실패: {e}")
        print("💡 토큰이 올바른지 확인해주세요.")
        return False

def main():
    """메인 함수"""
    success = setup_token_interactive()
    
    if success:
        print("\n🚀 이제 main.py를 실행할 수 있습니다:")
        print("   python main.py")
    else:
        print("\n⚠ 토큰 설정 없이 진행하면 일부 모델이 작동하지 않을 수 있습니다.")
        print("💡 대안 모델 사용을 고려해보세요:")
        print("   - BAAI/bge-base-en-v1.5 (캡션 임베딩)")
        print("   - Salesforce/blip-image-captioning-large (VLM)")

if __name__ == "__main__":
    main()
