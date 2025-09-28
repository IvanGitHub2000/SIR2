import argparse
import cv2
from ultralytics import YOLO

def detect_on_image(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image at: {image_path}")
        return

    results = model.predict(source=image, save=False, verbose=False)
    annotated_image = results[0].plot()

    cv2.imshow("Detection - Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_on_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Detection - Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_on_camera(model, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Detection - Camera", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Barcode Detection")
    parser.add_argument("--model", type=str, help="Path to YOLOv8 trained model")
    parser.add_argument("--image", type=str, help="Path to an image for detection")
    parser.add_argument("--video", type=str, help="Path to a video file for detection")
    parser.add_argument("--camera", action="store_true", help="Use webcam for real-time detection")

    args = parser.parse_args()
    model = YOLO(args.model)

    if args.image:
        detect_on_image(model, args.image)
    elif args.video:
        detect_on_video(model, args.video)
    elif args.camera:
        detect_on_camera(model)
    else:
        print("❌ You must provide either --image, --video or --camera")

#python detect_yolo.py --image "C:\\Users\\Ivan\\Desktop\\FAKULTET\\MASTER\\SIR1\\SIR1\\Image-bar-code-detection\\images\\barcode_01.jpg" --model "D:\\SIR2\\yolo\\runs\\detect\\train10\\weights\\best.pt" --za sliku
#python detect_yolo.py --video "C:\\Users\\Ivan\\Desktop\\FAKULTET\\MASTER\\SIR1\\Detekcija_Barkoda_Obradom_Slike\\detecting-barcodes-in-video\\video\\video_games.mov" --model "D:\\SIR2\\yolo\\runs\\detect\\train10\\weights\\best.pt" --za video
#python detect_yolo.py --camera --model "D:\\SIR2\\yolo\\runs\\detect\\train10\\weights\\best.pt" --za kameru


# from ultralytics import YOLO

# model = YOLO('D:\\SIR2\\BarBeR - Dataset\\yolo\\runs\\detect\\train10\\weights\\best.pt')  # učitaj sačuvani model

# results = model('C:\\Users\\Ivan\\Desktop\\FAKULTET\\MASTER\\SIR1\\SIR1\\Image-bar-code-detection\\images\\barcode_01.jpg')  # pokreni detekciju na slici

# for r in results:
#     r.show()  # Prikaz rezultata

# # Save result
    # output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    # cv2.imwrite(output_path, annotated_image)
    # print(f"✅ Detection saved at: {output_path}")