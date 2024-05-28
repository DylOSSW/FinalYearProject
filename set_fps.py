import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#cap.set(cv2.CAP_PROP_FPS, 10)

while True:
		ret, frame = cap.read()
		if not ret:
			break
		cv2.imshow('frame', frame)
		print(cap.get(cv2.CAP_PROP_FPS))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()
