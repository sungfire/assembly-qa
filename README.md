# Assembly QA Workspace

국회 정책 질의응답 서비스를 위한 풀스택 레포지터리입니다. FastAPI 기반 백엔드가 KoAlpaca LLM과 Firebase Admin을 사용해 답변과 대화 내역을 처리하며, React/Vite 프런트엔드가 Firebase Auth로 로그인한 사용자를 위한 채팅 UI를 제공합니다. 개발은 로컬 환경에서 진행하고, 실서비스용 백엔드는 RunPod 클라우드 GPU 노드에서 구동하는 시나리오를 가정합니다.

## 구성
- **backend/**: FastAPI 앱 (`app.py`)과 의존성 목록 (`requirements.txt`).
- **frontend/**: React/Vite SPA. Firebase 웹 SDK로 인증/세션을 관리하고 `/api` 엔드포인트로 백엔드와 통신합니다.
- **setup.sh**: 모델 다운로드, 배치, 파이썬 환경 구성을 자동화하는 스크립트. 로컬·RunPod에서 모두 재사용합니다.

## 선행 조건
- Linux 혹은 macOS 셸 환경
- `curl`, `unzip`, `rsync`, `python3`, `pip`
- Google Drive 아카이브를 받을 수 있는 네트워크 접근
- RunPod GPU 인스턴스(최소 16GB VRAM 권장) 및 CUDA 드라이버

## 로컬 개발 워크플로
1. 레포 클론 및 기본 세팅
   ```bash
   git clone <REPO_URL>
   cd assembly-qa
   ```
2. Firebase 서비스 계정 키(`serviceAccountKey.json`)를 `backend/` 디렉터리에 배치합니다.
3. 프런트엔드 환경 변수 설정 (`frontend/.env`).
4. 모델 파일을 로컬 머신에도 준비하려면 `bash setup.sh` 실행 (용량 20GB 이상 필요).
5. 프런트엔드 개발 서버
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   브라우저에서 http://localhost:5173 접속.

### `.env` 예시 (`frontend/.env`)
```dotenv
VITE_FIREBASE_API_KEY=AIza.......
VITE_FIREBASE_AUTH_DOMAIN=assembly-qa.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=assembly-qa
VITE_FIREBASE_STORAGE_BUCKET=assembly-qa.firebasestorage.app
VITE_FIREBASE_MESSAGING_SENDER_ID=1077862881933
VITE_FIREBASE_APP_ID=1:1077862881933:web:xxxxxxxxxxxxxxxxxxxxxx
VITE_FIREBASE_MEASUREMENT_ID=G-XXXXXXXXXX
# RunPod 백엔드 URL을 프록시로 사용할 경우
VITE_API_BASE_URL=https://<runpod-host>:8000/api
```

> 참고: 로컬에서는 백엔드를 띄우지 않고 프런트만 개발할 경우, 프록시나 원격 API URL을 사용하도록 Vite 설정을 조정하세요.

## RunPod 백엔드 배포
1. RunPod 인스턴스에 SSH 접속 후 레포 클론
   ```bash
   git clone <REPO_URL>
   cd assembly-qa
   ```
2. `backend/serviceAccountKey.json` 업로드 (scp 또는 RunPod 파일 매니저).
3. `setup.sh` 실행
   ```bash
   bash setup.sh
   ```
   - Google Drive에서 모델 아카이브 다운로드 → `/workspace/2.AI학습모델파일/1. 질의응답/nia15-polyglot-5.8b-koalpaca-v1.1b-qna-best/` 배치
   - `./.venv` 생성 후 `backend/requirements.txt` 설치
4. 백엔드 서버 기동
   ```bash
   source .venv/bin/activate
   cd backend
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
5. RunPod 포트 포워딩/Ingress 설정으로 8000번 포트를 외부에 노출합니다. (RunPod UI 또는 `pod_port_mapping` 활용)

## 운영 시 고려 사항
- Firestore 보안 규칙 및 서비스 계정 범위 재검토
- Firebase ID 토큰 검증 로직 도입(현재는 `uid`만 신뢰)
- 모델 경로를 환경 변수로 추출하여 환경별로 설정 가능하게 개선 권장
- GPU 메모리가 부족하면 모델 양자화나 추론 파라미터 튜닝 필요

## 테스트
- `curl`로 RunPod 백엔드 `/api/chat` 호출 후 응답 확인
- 프런트엔드에서 RunPod API endpoint를 가리키도록 Vite 환경 변수 (`VITE_API_BASE_URL` 등) 설정

## 라이선스
레포 내부 문서 및 모델 아카이브의 라이선스 조건을 확인한 뒤 서비스에 적용하십시오.
