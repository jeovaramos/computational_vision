import cv2
import copy
import numpy as np

# Webcam
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(
    history=15, dist2Threshold=50, detectShadows=False
)
mask_ref = False
frame_count = 0

while cap.isOpened():

    image = cap.read()[1]

    if not mask_ref:
        fgmask_ref = fgbg.apply(image)
        # image_ref = copy.copy(image)
        mask_ref = True

    if frame_count <= 501:
        frame_count += 1
        fgmask_ref += fgbg.apply(image)
        fgmask_mean = fgmask_ref / frame_count
        # image_ref += image
        # image_mean = image_ref / frame_count

    if frame_count == 500:
        print('Done')

    fgmask = fgbg.apply(image)
    # similarity = np.abs((fgmask - fgmask_mean).sum()/fgmask_mean.sum())
    # similarity = np.where(
    #     (np.stack([fgmask] * 3, axis=2) == 0) &
    #     (image != image_ref),
    #     1, 0
    # ).sum() / (image.size)

    similarity = np.where((fgmask > 0) & (fgmask_mean != fgmask), 0, 1).sum() / (fgmask.size)

    if similarity < 0.:
        print('Motion detected')

    label = f"{similarity * 100:.4f}"

    # if similarity < 0.9:
    #     label = f"Ok! {similarity:.4f}"
    # else:
    #     label = f"Moved! {similarity:.4f}"

    cv2.putText(
            img=image,
            text=label,
            org=(50, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2
        )

    cv2.imshow('frame', image)
    cv2.imshow('frame masked', fgmask)
    cv2.imshow('frame mean', fgmask_mean)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
