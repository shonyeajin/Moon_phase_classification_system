# Moon_phase_classification_system 🌛
**Moon Phase Classification System** source code archive
  
## Installation & Data preparation
  - Clone this repository
  
  - Organize them as following:
    ```
    Moon_phase_classification_system/
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
 1. 달 데이터의 평균 및 표준편차를 이용한 시각화 -> avg_std_plot.py 실행하기
 2. VGG-16, VGG-19, ResNet50, Inception V3, DenseNet121, NasNetLarge, MobileNetV3Large, (Our proposal) VGG16 + VGG19, (Our proposal) VGG16 + VGG19 + DenseNet121 모델들을 build, train, validate, test -> train.py 실행하기
  

## 제안 방법
![제안방법](https://user-images.githubusercontent.com/55689863/190143267-a8011093-4ecd-4db1-8ba7-610dcd895430.png)

## 개발환경
colab
