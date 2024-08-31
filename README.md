# Korean Conversation Context Inference
- [인공지능(AI)말평 대화맥락추론(가 유형)](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=144&clCd=END_TASK&subMenuId=sub01) 제출코드
- [모델링크](https://huggingface.co/LostCow/POLAR-14B-dialogue-inference)

## 실행방법
### 환경설정
```
pip install --upgrade pip
pip install -r requirements.txt
```
### 데이터 다운로드
- 편의상 데이터 이름을 train.json, dev.json, test.json으로 변경함
```
https://kli.korean.go.kr/taskOrdtm/taskDownload.do?taskOrdtmId=144&clCd=END_TASK&subMenuId=sub02
```

### 데이터 증강
```
$ bash scripts/run_make_aug.sh
```

### 모델 학습
```
$ bash scripts/run_train.sh
```

### 모델 추론
```
$ bash scripts/run_infer.sh
```