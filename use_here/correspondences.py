import cv2
import json
import numpy as np

# Ölçek faktörü (örneğin %50)
scale = 0.5

# Nokta listeleri
points_img1 = []
points_img2 = []
mode = [0]  # 0: img1, 1: img2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Orijinal koordinatlara geri ölçekle
        x_scaled = int(x / scale)
        y_scaled = int(y / scale)
        if mode[0] == 0:
            points_img1.append((x_scaled, y_scaled))
            print(f"img1_xy: {x_scaled}, {y_scaled}")
            mode[0] = 1
        else:
            points_img2.append((x_scaled, y_scaled))
            print(f"img2_xy: {x_scaled}, {y_scaled}")
            mode[0] = 0

# Görüntüleri oku
img1 = cv2.imread("images/test_1_1.jpeg")
img2 = cv2.imread("images/test_1_2.jpeg")

# Yan yana birleştir
h = max(img1.shape[0], img2.shape[0])
combo = np.zeros((h, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
combo[:img1.shape[0], :img1.shape[1]] = img1
combo[:img2.shape[0], img1.shape[1]:] = img2

# Küçültülmüş gösterim
combo_display = cv2.resize(combo, (0, 0), fx=scale, fy=scale)

cv2.namedWindow("Select Correspondences (scaled)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Correspondences (scaled)", mouse_callback)

print("Sol resimde (img1) bir nokta seç, sonra sağ resimde (img2) karşılığını seç. ESC ile çık.")

while True:
    vis = combo_display.copy()

    for p in points_img1:
        cv2.circle(vis, (int(p[0] * scale), int(p[1] * scale)), 4, (0, 255, 0), -1)
    for p in points_img2:
        p2 = (int((p[0] + img1.shape[1]) * scale), int(p[1] * scale))
        cv2.circle(vis, p2, 4, (255, 0, 0), -1)

    cv2.imshow("Select Correspondences (scaled)", vis)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

# JSON olarak kaydet
if len(points_img1) == len(points_img2):
    correspondences = [
        {
            "img1_xy": list(p1),
            "img2_xy": list(p2)
        }
        for p1, p2 in zip(points_img1, points_img2)
    ]

    with open("correspondences.json", "w") as f:
        json.dump(correspondences, f, indent=4)
    print("✅ correspondences.json kaydedildi.")
else:
    print("❌ Her img1 noktası için bir img2 noktası tıklanmalı!")
