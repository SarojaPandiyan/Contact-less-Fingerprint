import cv2
import fingerprint_enhancer

image=cv2.imread("F:\Sem_3\Design Thinking\Implementation\dataset_1\dataset\\real_data\mysterychap.jpg",0)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # larger than the width of the widest ridges
low = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)  # locally lowest grayvalue
high = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)  # locally highest grayvalue
image = (image - low) / (high - low + 1e-6)
out = fingerprint_enhancer.enhance_Fingerprint(image)
cv2.imwrite('F:\Sem_3\Design Thinking\Implementation\\new_dataset\\mysterychap.bmp',out)

name=input("\nENTER NAME ")
with open('F:\Sem_3\Design Thinking\Implementation\db.csv', mode ='+a')as file:
    file.write(f"\n{name},F:/Sem_3/Design Thinking/Implementation/new_dataset/mysterychap.bmp")
print("\nFingerprint registered successfully")
file.close()

