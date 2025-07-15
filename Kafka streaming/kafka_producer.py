import cv2
import json
import time
import os
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f"[ERROR] Message delivery failed: {err}")
    else:
        print(f"[INFO] Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_video(video_path, bootstrap_servers='localhost:9092', topic='video-stream', resize_factor=0.5, quality=50):
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'video-producer',
        'message.max.bytes': 10000000  # 10MB
    }
    producer = Producer(conf)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot read video file")
        return

    filename = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[INFO] End of video or frame is empty.")
            break

        if resize_factor < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        frame_bytes = buffer.tobytes()

        message = {
            'frame_id': frame_count,
            'timestamp': time.time(),
            'data': frame_bytes.hex(),
            'width': frame.shape[1],
            'height': frame.shape[0],
            'filename': filename
        }

        try:
            producer.produce(
                topic=topic,
                value=json.dumps(message).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)
        except Exception as e:
            print(f"[ERROR] Failed to produce message: {e}")
            continue

        frame_count += 1
        time.sleep(0.01)

    # Send end-of-stream marker
    producer.produce(
        topic=topic,
        value=json.dumps({'frame_id': -1, 'timestamp': time.time(), 'data': None}).encode('utf-8'),
        callback=delivery_report
    )

    producer.flush()
    cap.release()
    print("[INFO] Video stream finished.")

if __name__ == "__main__":
    video_path = "test/IMG_0418.MOV"
    produce_video(video_path, resize_factor=0.5, quality=50)