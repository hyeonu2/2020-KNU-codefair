# 2020-KNU-codefair  
### 이제는 CCTV도 알고리즘이다! : 야간 및 주간에 최적화 된 Super CCTV 
### 팀명 : HBU(호반우)

주간모드 코드와 야간모드 코드로 구별되어 있습니다.
모든 코드는 파이썬으로 작성되었으며, Open CV를 기반으로구현했습니다.

![프레젠테이션1](https://user-images.githubusercontent.com/54931018/99875100-56213000-2c30-11eb-831b-60595e7849fe.png)


# 기존의 문제점 및 발전 방안

## 문제점 
 최근 공공안전과 시설보호 및 군사적 용도로 CCTV가 널리 쓰이고 있다. 우리나라는 세계에서도 손꼽히는 CCTV 대국으로서 최근 5년간 그 설치대수가 매년 평균 10만대 이상 증가하고 있는 추세이다. 하지만 CCTV의 성능이나 질은 그리 높아지지 않았다. 서울지하철 역사와 전동차 안에 설치된 CCTV의 95%가 50만 화소 미만의 저화질 카메라이다.(2018년 기준) 
 이러한 카메라들은 빛이 충분한 상태에서도 멀리 있는 사물은 물론이고 가까이에 있는 사물도 정확하게 식별하지 못해 범죄 수사 등에 별다른 도움을 주지 못한다. 특히 조명이 거의 없는 야간에 CCTV의 역할은 주간보다 더욱 중요하지만, 대부분의 CCTV는 야간에 식별이 거의 어려운 영상정보만을 가지고 있다. 
 CCTV 카메라는 감시 역할을 온전히 수행하기위해 어떠한 공간, 조명 상황에서도 충분한 영상 정보를 제공해야 한다. 지금까지의 CCTV는 성능 향상을 위해 LED 보조램프, 고해상도 및 고감도 이미지 센서 등 다양한 기술들을 활용하고 있다. 하지만 이러한 방법들은 고비용이 들고 추가적인 장치가 필요하기에 구현성 및 활용성 면에서 불리한 단점이 있다. 

## 발전 방안 
 우리는 카메라에 실시간 SW 알고리즘을 설계 및 탑재함으로써, 동일 H/W대비 CCTV의 성능을 극대화하고자 한다. 알고리즘을 주변 광량에 따라 주간 상태와 야간 상태로 나누어 구성함으로써 주간/야간 각각에 최적화 된 Super CCTV를 구현하고자 한다. 이를 통해 SW 변경만으로 우수한 시각 정보를 가진 영상을 제공함으로써, 다양한 분야에 고품질/저비용의 CCTV의 대중화를 이루고자한다.


# 세부내용(구현방법 등)
 주간모드개발(안용현 팀원)과 야간모드개발(이현우 팀장)으로 나뉘어 진행했다. 
 
 주간 모드 알고리즘은 HDR 영상 합성을 통해 CCTV의 동적 범위를 확장함으로써 어두운 영역과 밝은 영역에 대한 가시성과 색 표현력을 향상시킨다. 노출이 다른 2장의 LDR 영상의 배경 정보와 세부 정보를 각각 다른 알고리즘으로 합성하기 위해 Base layer과 세부 층 Detail layer으로 분리한다. 2장의 LDR 영상을 정확히 같은 시각에 얻기 위해 CMOS 멀티센서카메라를 사용하였다. 하드웨어 상의 화각 차이를 보상하기 위해Homography Matching과 Retinex 기반의 정합 알고리즘을 사용하였다. 이후 2 장의 노출 영상에 Saliency map과 Saturation map을 계산하여 Base 합성을 위한 가중치 맵을 생성한다. 이후 고 노출 영상에 대해 밝기에 따른 Local Gamma Compensation을 통한 전처리 과정을 거쳐 두 LDR영상을 합성한다. 최종적으로 합성영상의 휘도 채널과 원 영상의 휘도 채널의 비를 이용해 채도 보상 및 샤프닝 마스킹을 통해 영상의 채도와 선명도를 향상시킨다.
        
 야간모드 알고리즘은 가시광 영상과 근적외선 영상의 합성을 통해 영상의 세부 정보를 포함함과 동시에 정확한 색채정보를 가진 영상을 합성한다. 먼저 나란히 나열된 가시광 카메라와 근적외선 카메라를 통해 가시광 영상과 근적외선 영상을 얻는다. 두 카메라의 시야각 차이로 인하여 경계의 불일치성을 보상하고자 호모그래피 매칭과 레티넥스 알고리즘 기반의 정합 알고리즘을 사용한다.
이후 웨이블릿 변환을 이용해 각 영상의 휘도 채널을 세부 성분과 기저 성분으로 분해한 후, 가중치 맵을 이용하여 영상 합성을 진행한다. 이 과정의 결과로 합성되는 영상의 휘도 채널은 두 입력 영상의 세부 성분 중 우수한 영역만을 모아서 합성된다.
두 영상의 휘도 채널을 합성에 의해 입력대비 출력 휘도가 변화할 경우, 채도 채널과의 불균형이 발생해 색 표현이 부자연스러워진다. 따라서 가시광 영상의 채도 채널에 대해 합성 휘도 채널과 가시광 영상의 휘도 채널의 비를 이용해 채도 보상을 실행한다. 마지막으로 합성 영상의 명도 성분에 샤프닝 마스크를 적용함으로 영상의 선명도 개선을 수행한다.

 주간 상태 모드에서는 고노출 및 저노출 영상을 합성하여 영상의 시각 동적 범위를 확장함으로써 어두운 영역과 밝은 영역에 대한 가시성과 색 표현력을 향상시켰다. 또한 과포화 현상 및 Blurring 현상을 방지할 수 있었다. 야간 모드에서는 가시광과 IR 영상을 합성하여 어두운 상황 속에서도 CCTV 영상의 디테일 및 베이스 정보를 잘 식별할 수 있다. 주간 및 야간 모드 알고리즘을 통해 얻은 최종 결과 영상은 색 및 휘도 표현이 우수하고 사물 인식력이 대폭 향상된다. 또한 특수 센서를 사용하지 않기에 활용도가 높고 비용 측면에서 큰 강점을 보인다.  주간모드와 야간모드 각각에 대한 알고리즘 개발 및 검증을 완료했으며, 현재 주간모드와 야간모드 알고리즘을 결합한 후, 주변 광량에 따라 주간 및 야간 모드를 선택적 또는 적응적으로 작동하는 Super CCTV를 구현하는 중이다.


# 기대효과

고성능의 SW기반 Super CCTV를 저렴한 비용으로 구현할 수 있을 것이고, 이를 통해 범죄예방, 군사감시, 시설보호 등 다양한 분야에서 CCTV의 대중화를 이룰 수 있다. 더불어 최근 활발히 연구되고있는 지능형 CCTV나 아마존과 같은 무인 마켓 등에서 사물의 세부 정보를 얻기위해사 사용되는 AI CCTV에 접목한다면, Detection/Classification의 정확도를 대폭 향상시키는 전처리 과정으로 역할 수 있을 것으로 기대한다.
