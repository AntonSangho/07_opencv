# 3채널 컬러 영상은 하나의 색상을 위해서 24비트(8x3) 
# 16777216가지의 색상을 표현 가능

# 모든 색을 다 사용하지 않고 비슷한 그룹의 색상을 지어서 같은 색상으로 처리
# 처리 용량 간소화 

import numpy as np
import cv2

#K = 16 # 군집화 개수
K = 8 # 군집화 개수
img = cv2.imread('../img/taekwonv1.jpg')
data = img.reshape((-1, 3)).astype(np.float32) 
# 데이터 평균을 구할때 수소점 이하값을 가질 수있으므로 변환
# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클리스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환 
center = np.uint8(center)
print(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]
# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Kmeans color', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()