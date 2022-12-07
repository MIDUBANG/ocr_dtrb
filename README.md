# ocr_dtrb
deep-text-recognition-benchmark를 이용한 OCR 사용자모델 학습

dependency : torch, lmdb, torchvision, nltk, natsort, pillow, requests, opencv-python, tqdm, fire (use `pip install [library]` command)

# 1. 문장 데이터 생성

### TextRecognitionDataGenerator
- 다양한 노이즈를 포함하는 문장 생성을 돕는 오픈소스
- 단어 데이터셋에서 무작위로 10개의 단어를 뽑아 띄어쓰기 포함하여 조합하도록 하였다.
- 다음과 같은 종류의 문장 생성
1) 기본 9천개
2) 배경 조정 3천개
3) 기울기 조정 9천개
4) 블러 조정 3천개
5) 왜곡 조정 3천개

### 실행 커맨드
```bash
$ cd data/generator/TextRecognitionDataGenerator
$ sh generate_data_5type_test.sh
```
위 커맨드 실행시 ocr_dtrb/data/generator/TextRecognitionDataGenerator/out 경로에 5가지 유형의 데이터가 생성된다.

<br>

### 문장 생성에 사용한 데이터
1.  단어 데이터 
- 국립국어원 한국어 학습용 어휘목록(단어 5965개, 글자 974개)
-  법무부 생활법률지식 데이터 (단어 287개)

2. 폰트 데이터 
- 네이버 나눔글꼴 23종
- 네이버 나눔손글씨 109종

<br>

### 데이터 생성 디렉토리 구조
```bash
/TextRecognitionDataGenerator/out
     ├── basic
     │     ├── [이미지 파일명]
     │     └──  ...
     ├── skew   
     │     ├── [이미지 파일명]
     │     └──  ...
     ├── blur   
     │     ├── [이미지 파일명]
     │     └──  ...
     ├── back   
     │     ├── [이미지 파일명]
     │     └──  ...
     └── dist   
            ├── [이미지 파일명]
            └──  ...

```
<br>
<br>

### 문장 데이터 생성시 쓰이는 옵션
generate_data_5type.sh에서 편집 가능

- `--c`: count(문장 이미지 개수)
- `--w`: length
- `--f`: format (the height of the produced images)
- `--l`: language
- `--k`: skew angle
- `--rk`: when set, the skew angle will be randomized between the value set with -k and it's opposite
- `--d`: distorsion
- `--do`: distorsion_orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both
- `--bl`: blur
- `--b`: background
- `output_dir`: the output directory

<br>
<br>

### 생성된 문장 데이터 예시
1. basic

<br>
 
![](https://velog.velcdn.com/images/goinggoing/post/54717ea5-b54c-4b0e-8061-2b0c1a540602/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/7763c6f8-7ffb-4108-bb4f-e8c0f3b8215c/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/6635063a-b449-40eb-88ab-e97510143a9b/image.jpg)

<br>
 
2. skew

<br>
 
![](https://velog.velcdn.com/images/goinggoing/post/8afe2399-7448-4298-a2ae-427aff781451/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/f203d55a-f07c-466a-8e4d-b36e899b20c3/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/af0760b7-8201-464c-a63a-7b2c4ca3de3d/image.jpg)

<br>
 
3. blur

<br>
 
![](https://velog.velcdn.com/images/goinggoing/post/8da1ae16-51ca-4d1d-8b19-9c017be30006/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/fb0d684d-bed4-47ea-9035-1819e4987e67/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/b56baed9-2691-4893-b929-5ec1008d3ee6/image.jpg)

<br>
 
4. back

<br>
 
![](https://velog.velcdn.com/images/goinggoing/post/54e9354f-6697-412d-b901-da09cf5b5121/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/3fb22cfc-cc3e-4502-9c2c-57916a012f4d/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/7a70d9bb-3645-4400-a3cf-75c6c91c48e7/image.jpg)

<br>
 
5. dist

<br>
 
![](https://velog.velcdn.com/images/goinggoing/post/f4270b22-82a2-4aab-9a89-2d8867a1fec5/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/9a38baf7-90ad-4d6d-a1e0-d4af9b601006/image.jpg)
![](https://velog.velcdn.com/images/goinggoing/post/8a6b967c-7816-4c27-96c4-e5c1ac890318/image.jpg)

<br>
 
<br>
 


# 2. gt file 생성
문장 유형별(basic, back, blur, skew, dist) gt file을 생성한다.

### 커맨드

```bash
$ ./create_gt_file.sh basic | tee -a gt_basic.txt
$ ./create_gt_file.sh back | tee -a gt_back.txt
$ ./create_gt_file.sh blur | tee -a gt_blur.txt
$ ./create_gt_file.sh skew | tee -a gt_skew.txt
$ ./create_gt_file.sh dist | tee -a gt_dist.txt
```
<br>

### 생성된 gt file 내 포맷
gt file의 문장 포맷 : [filedir]\t[label]  (\t으로 구문되며 문장 끝에는 \n)

예시:
![](https://velog.velcdn.com/images/goinggoing/post/fbd04cef-1777-4896-9cb7-34aca8446f09/image.png)



<br>

# 3. lmdb 포맷으로 데이터셋 생성


### 커맨드
문장유형별(basic, back, blur, skew, dist)로 5번 반복한다. 문장 유형에 따라`--gtFile`,`--outputPath` 변경하여 사용한다.



```bash
$ cd ../../..
$ python3 data/create_lmdb_dataset.py 
	--inputPath data/generator/TextRecognitionDataGenerator/
    --gtFile data/generator/TextRecognitionDataGenerator/gt_basic.txt  
    --outputPath data/data_lmdb_release/training/basic;
```
```bash
$ python3 data/create_lmdb_dataset.py 
	--inputPath data/generator/TextRecognitionDataGenerator/
    --gtFile data/generator/TextRecognitionDataGenerator/gt_skew.txt  
    --outputPath data/data_lmdb_release/training/skew;
```
```bash
$ python3 data/create_lmdb_dataset.py 
	--inputPath data/generator/TextRecognitionDataGenerator/
    --gtFile data/generator/TextRecognitionDataGenerator/gt_back.txt  
    --outputPath data/data_lmdb_release/validation/back;
```
```bash
$ python3 data/create_lmdb_dataset.py 
	--inputPath data/generator/TextRecognitionDataGenerator/
    --gtFile data/generator/TextRecognitionDataGenerator/gt_blur.txt  
    --outputPath data/data_lmdb_release/validation/blur;
```
```bash
$ python3 data/create_lmdb_dataset.py 
	--inputPath data/generator/TextRecognitionDataGenerator/
    --gtFile data/generator/TextRecognitionDataGenerator/gt_dist.txt  
    --outputPath data/data_lmdb_release/validation/dist;
```


<br>
<br>

### 실행 결과 디렉토리 구조
다음과 같은 디렉토리 구조가 형성된다.
```bash
/data_lmdb_release
├── /training
│     ├── basic
│     │     ├── data.mdb
│     │     └── lock.mdb
│     └── skew   
│            ├── data.mdb
│            └── lock.mdb
└── /validation
       ├── back
       │     ├── data.mdb
       │     └── lock.mdb
       ├── blur
       │     ├── data.mdb
       │     └── lock.mdb
       └── dist
              ├── data.mdb
              └── lock.mdb
```

<br>

# 4. 사용자 모델 학습(training)
여기서부턴 gpu 환경에서만 가능하다.

### 커맨드
```bash
$ cd deep-text-recognition-benchmark
$ CUDA_VISIBLE_DEVICES=0 python3 train.py 
  --train_data ../data/data_lmdb_release/training 
  --valid_data ../data/data_lmdb_release/validation 
  --select_data basic-skew 
  --batch_ratio 0.5-0.5 
  --Transformation TPS 
  --FeatureExtraction VGG 
  --SequenceModeling BiLSTM 
  --Prediction CTC 
  --data_filtering_off  
  --valInterval 100 
  --batch_size 128 
  --batch_max_length 50 
  --workers 6 
  --distributed 
  --imgW 400;
```
<br>

### 학습 옵션
train.py의 옵션을 커스텀해 학습 가능하다.
- `--train_data` : path to training dataset
- `--valid_data` : path to validation dataset
- `--select_data`: directories to use as training dataset(default = 'basic-skew')
- `--batch_ratio` 
- `--Transformation` : choose one - None|TPS
- `--FeatureExtraction`: choose one - VGG|RCNN|ResNet 
- `--SequenceModeling`: choose one - None|BiLSTM
- `--Prediction` : choose one - CTC|Attn
- `--data_filtering_off` : skip data filtering when creating LmdbDataset
- `--valInterval` : Interval between each validation
- `--workers` :  number of data loading workers
- `--distributed`
- `--imgW` : the width of the input image
- `--imgH` : the height of the input image

<br>

### 학습 결과
- ocr_dtrb/deep-text-recognition-benchmark/saved_models 디렉토리에 학습시킨 모델별 `log_train.txt`, `best_accuracy.pth`, `best_norem_ED.pth` 파일이 저장된다. 
- log_train.txt에서는 iteration마다 best_accuracy와 loss 값이 어떻게 변하는지 확인 가능하다.
![](https://velog.velcdn.com/images/goinggoing/post/3d5391c9-7d6f-48fd-b4e6-cda1d13ddf61/image.png)
- best_accuracy.pth 파일을 이용해 evaluation과 demo가 가능하다. 

<br>

# 5. 사용자 모델 테스트(evaluation)


- 본 학습에서는 training data : validation data = 2:1 비율로 설정했기 때문에 test data 생성과 테스트 과정을 생략했다. 
- test 과정을 진행하고 싶다면 1~3단계에서 테스트 데이터도 생성/가공하면 된다.
  
### 커맨드
```shell script
$ CUDA_VISIBLE_DEVICES=0 python3 test.py 
  --eval_data ../data/data_lmdb_release/evaluation 
  --benchmark_all_eval 
  --Transformation TPS 
  --FeatureExtraction VGG 
  --SequenceModeling None 
  --Prediction CTC 
  --saved_model saved_models/Test-TPS-VGG-None-CTC-Seed/best_accuracy.pth 
  --data_fil1tering_off 
  --workers 2 
  --batch_size 128 
  --imgW 400;
```
- 위 커맨드는 테스트 문장데이터로 lmdb 데이터셋을 생성하여 data_lmdb_release/evaluation 경로로 저장했다고 가정했다. 
- 가장 정확도가 높았던 학습 모델인 Test-TPS-VGG-None-CTC-Seed를 테스트에 사용했다. 다운로드 받아 사용해볼 수 있다. (용량이 커 구글 드라이브로 첨부) [Test-TPS-VGG-None-CTC-Seed](https://drive.google.com/file/d/16JvCdkkEKum7CaFH4TkAVu1YWMC3NQ9_/view?usp=sharing)
- 직접 학습시켜 새롭게 저장된 모델도 사용할 수 있다.

<br>

### 테스트 옵션
학습 시에 사용한 옵션들을 거의 동일하게 사용할 수 있다. 
- `--eval_data` : path to evaluation dataset
- `--benchmark_all_eval` : evaluate 3 benchmark evaluation datasets
- `--saved_model` : path to saved_model to evaluation

<br>

# 6. 학습 모델 데모

- `--Transformation`, `--FeatureExtraction`, `--SequenceModeling`, `--Prediction` 옵션을 이용해 각 스테이지에서 사용할 모듈을 결정한다. 
- 학습 시에 같은 모듈을 사용했더라도 설정한 옵션에 따라 accuracy와 loss가 다를 수 있다. 학습한 모델 중 데모를 시도할 모델은 `--saved_model` 옵션으로 지정할 수 있다.
- `--image_folder` 옵션으로 데모 쓰일 디렉토리 경로를 지정한다.

<br>

### 커맨드
```shell script
$ CUDA_VISIBLE_DEVICES=0 python3 demo.py 
  --Transformation TPS   
  --FeatureExtraction VGG   
  --SequenceModeling None   
  --Prediction CTC  
  --image_folder ../data/demo_image   
  --saved_model saved_models/Test-TPS-VGG-None-CTC-Seed/best_accuracy.pth;
```
- saved_models 디렉토리에 학습시킨 모델 중 가장 정확도 높았던 모델을  다운로드 받아 사용해볼 수 있다. (용량이 커 구글 드라이브로 첨부) [Test-TPS-VGG-None-CTC-Seed](https://drive.google.com/file/d/16JvCdkkEKum7CaFH4TkAVu1YWMC3NQ9_/view?usp=sharing)
- 예시는 saved_models/Test-TPS-VGG-None-CTC-Seed 디렉토리를 만들고 위의 모델을 다운 받아 이용한 데모이다. saved_models 디렉토리에 저장되는 학습 모델 경로로 지정하면 직접 학습시킨 다른 모델로도 가능하다.

- 데모를 위해 나눔고딕, 맑은 고딕, 굴림 폰트가 사용된 문장 이미지 데이터를 data/demo_image 디렉토리에 첨부해두었다. 다른 문장 이미지로도 가능하다. 




<br>

# Acknowledgements
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) , [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)
