import numpy as np, cv2

# 전역 변수
roi_points = []
roi_selected = False

def mouse_callback(event, x, y, flags, param):
    """마우스 콜백 함수 - ROI 영역 선택"""
    global roi_points, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        roi_selected = False
        
    elif event == cv2.EVENT_MOUSEMOVE and len(roi_points) == 1:
        # 드래그 중인 사각형 표시
        img_copy = param.copy()
        cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Select ROI', img_copy)
        
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        roi_selected = True
        
        # 선택된 ROI 표시
        img_copy = param.copy()
        cv2.rectangle(img_copy, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.putText(img_copy, 'Press SPACE to analyze or ESC to exit', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Select ROI', img_copy)

def analyze_roi_colors():
    """ROI 영역의 색상 분석"""
    global roi_points, roi_selected
    
    # 이미지 읽기
    img = cv2.imread('../img/load_line.jpg')
    
    if img is None:
        print("Image not found. Please check the file path.")
        return
    
    # 이미지 크기 조정
    img = cv2.resize(img, (800, 600))
    
    print("Instructions:")
    print("1. Click and drag to select ROI area")
    print("2. Press SPACE to analyze selected area")
    print("3. Press ESC to exit")
    print("-" * 50)
    
    # 윈도우 생성 및 마우스 콜백 설정
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', mouse_callback, img)
    
    # 초기 이미지 표시
    cv2.imshow('Select ROI', img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and roi_selected:  # SPACE키로 분석 실행
            # ROI 영역 추출
            x1, y1 = roi_points[0]
            x2, y2 = roi_points[1]
            
            # 좌표 정렬 (왼쪽 위, 오른쪽 아래)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            roi = img[y_min:y_max, x_min:x_max]
            
            if roi.size > 0:
                analyze_roi_region(roi, x_min, y_min, x_max, y_max)
            else:
                print("Invalid ROI selection. Please try again.")
                
        elif key == 27:  # ESC키로 종료
            break
    
    cv2.destroyAllWindows()

def classify_color_hsv(h, s, v):
    """HSV 값을 기반으로 색상 분류"""
    
    # 흰색: 낮은 채도, 높은 명도
    if s < 30 and v > 180:
        return "White(Lane)"
    
    # 노란색: 색조 15-35, 높은 채도, 중간 이상 명도
    elif 15 <= h <= 35 and s > 80 and v > 100:
        return "Yellow(Lane)"
    
    # 어두운 아스팔트: 낮은 명도
    elif v < 80:
        return "Dark(Asphalt)"
    
    # 회색 아스팔트: 낮은 채도, 중간 명도
    elif s < 50 and 80 <= v <= 180:
        return "Gray(Asphalt)"
    
    # 기타
    else:
        return "Other"

def analyze_roi_region(roi, x1, y1, x2, y2):
    """선택된 ROI 영역 색상 분석 (HSV 사용)"""
    
    # BGR을 HSV로 변환
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 데이터 준비 (3차원 → 2차원)
    data = roi_hsv.reshape((-1, 3)).astype(np.float32)
    
    # K-means 설정
    K = 4  # 클러스터 수를 4개로 증가
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    # K-means 클러스터링 적용
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 결과 출력
    print("\n" + "="*70)
    print(f"HSV ROI Analysis Results - Area: ({x1},{y1}) to ({x2},{y2})")
    print("="*70)
    
    # 클러스터를 픽셀 수 기준으로 정렬
    cluster_info = []
    for i in range(K):
        pixel_count = np.sum(label.ravel() == i)
        percentage = pixel_count / len(label.ravel()) * 100
        cluster_info.append((i, pixel_count, percentage, center[i]))
    
    # 픽셀 수 기준으로 정렬 (많은 순서대로)
    cluster_info.sort(key=lambda x: x[1], reverse=True)
    
    for idx, (i, pixel_count, percentage, hsv_center) in enumerate(cluster_info):
        h, s, v = hsv_center
        
        # HSV 기반 색상 분류
        color_type = classify_color_hsv(h, s, v)
        
        print(f"Cluster {i}: {color_type:15s} | HSV({h:3.0f}, {s:3.0f}, {v:3.0f}) | {percentage:5.1f}%")
        
        # BGR 값도 참고용으로 표시
        # HSV를 BGR로 변환하여 표시
        hsv_sample = np.uint8([[[h, s, v]]])
        bgr_sample = cv2.cvtColor(hsv_sample, cv2.COLOR_HSV2BGR)[0][0]
        b, g, r = bgr_sample
        print(f"         -> Equivalent BGR({b:3d}, {g:3d}, {r:3d})")
    
    print("="*70)
    print("HSV Color Classification Guide:")
    print("- White(Lane):    Low Saturation(<30), High Value(>180)")
    print("- Yellow(Lane):   Hue(15-35), High Saturation(>80), Medium+ Value(>100)")  
    print("- Dark(Asphalt):  Low Value(<80)")
    print("- Gray(Asphalt):  Low Saturation(<50), Medium Value(80-180)")
    print("="*70)
    print("Press SPACE to analyze again or ESC to exit")

# 실행
if __name__ == "__main__":
    analyze_roi_colors()