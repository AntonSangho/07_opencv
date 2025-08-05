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

def analyze_roi_region(roi, x1, y1, x2, y2):
    """선택된 ROI 영역 색상 분석"""
    
    # 데이터 준비 (3차원 → 2차원)
    data = roi.reshape((-1, 3)).astype(np.float32)
    
    # K-means 설정
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # K-means 클러스터링 적용
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"ROI Analysis Results - Area: ({x1},{y1}) to ({x2},{y2})")
    print("="*60)
    
    for i in range(K):
        b, g, r = center[i]
        pixel_count = np.sum(label.ravel() == i)
        percentage = pixel_count / len(label.ravel()) * 100
        
        # 색상 유형 판별
        #if b > 180 and g > 180 and r > 180:
        if b > 200 and g > 200 and r > 200:
            color_type = "White(Lane)"
        elif g > 150 and r > 150 and b < 100:
            color_type = "Yellow(Lane)"
        elif b < 80 and g < 70 and r < 40:
            color_type = "Dark(Asphalt)"
        else:
            color_type = "Other"
            
        print(f"Cluster {i}: {color_type:12s} | BGR({b:3.0f}, {g:3.0f}, {r:3.0f}) | {percentage:5.1f}%")
    
    print("="*60)
    print("Press SPACE to analyze again or ESC to exit")

# 실행
if __name__ == "__main__":
    analyze_roi_colors()