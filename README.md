# Moon_phase_classification_system 🌛
**Moon Phase Classification System** source code archive
  
## Installation & Data preparation
  - Clone this repository
  
  - Organize them as following:
    ```
    Contaminant_Discrimination/
      └── data
          ├── last quarter moon/
          ├── full moon/
          ├── first quarter moon/
          ├── dark moon/
          └── crescent moon/
    ```
    
  - Download dataset from
      - [last quarter moon](https://drive.google.com/drive/folders/1LgDoeIt_kELMCbaR9OfY83kyUOKZLzHs?usp=sharing)
      - [full moon](https://drive.google.com/drive/folders/1uT1CvklBNQsKLx3hdADOh9FpYmFG0va4?usp=sharing)
      - [first quarter moon](https://drive.google.com/drive/folders/1q7vMkgyT_2N38aZspgYX76Q4hXjRNIKl?usp=sharing)
      - [dark moon](https://drive.google.com/drive/folders/1P4Yd67f4f73ELwiZMCOjdJU6E6fsq7Zs?usp=sharing)
      - [crescent moon](https://drive.google.com/drive/folders/1Dy_VH3L4NC7pPY6NHZW652OkypiJFqyj?usp=sharing)
   


## Run
 1. How to train
    - data.yaml 'train', 'val', 'test' 경로 수정하기
    - cd yolov5
    - python train.py --img 416 --batch 16 --epoch 50 --data ../data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name [결과 파일 이름]

 2. 크롭 이미지 생성하기 -> crop.py 실행
 3. 유사도 비교하기 -> similarity.py 실행
 4. 데이터 분포 확인하기 -> eda.py 실행
  

## 제안 방법
![제안방법](https://user-images.githubusercontent.com/55689863/189947587-5b2276d5-a5a6-4361-b601-7aed400c2032.png)

## 개발환경
colab
