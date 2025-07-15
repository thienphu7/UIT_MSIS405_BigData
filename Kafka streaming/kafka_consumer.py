import cv2
import json
import numpy as np
import os
from confluent_kafka import Consumer, KafkaError
from ultralytics import solutions

def consume_and_process(bootstrap_servers='localhost:9092', topic='video-stream', group_id='video-consumer-group'):
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'fetch.message.max.bytes': 10000000,
        'max.partition.fetch.bytes': 10000000
    }
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    video_writer = None
    fps = 30

    region_points = [
        (1471.517, 0),
        (1443.904, 57.112),
        (1394.524, 144.476),
        (1335.988, 253.086),
        (1248.512, 411.679),
        (1119.008, 664.16),
        (999.711, 905.296),
        (807.047, 1288.892),
        (555.329, 1825.708),
        (392.973, 2157.952),
        (3839, 2159),
        (3839, 1745.031),
        (3651.738, 1489.924),
        (3338.006, 1068.123),
        (3034.179, 638.542),
        (2723.031, 196.924),
        (2603.093, 0),
        (1471.517, 0)
    ]

    # Region polygon of IMG_0410.MOV (float)
    """
    region_points = [
        (916.348, 0.667),
        (975.51, 90.097),
        (1005.758, 156.779),
        (1039.544, 242.57),
        (1071.494, 361.367),
        (1088.019, 523.108),
        (1097.788, 685.614),
        (1089.973, 806.031),
        (1065.874, 975.134),
        (1034.611, 1143.478),
        (953.197, 1463.442),
        (841.497, 1821.317),
        (707.326, 2159),
        (3839, 2159),
        (3839, 927.766),
        (3740.625, 842.546),
        (3549.262, 683.997),
        (3465.744, 617.257),
        (3364.419, 544.068),
        (3142.08, 373.274),
        (2837.102, 183.319),
        (2747.188, 126.539),
        (2509.927, 0),
        (916.348, 0.667)
    ]
    """

    trackzone = solutions.TrackZone(
        show=True,
        region=region_points,
        model="my_model.pt",
    )

    print("Starting Kafka consumer...")
    message_count = 0

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"[ERROR] Kafka consumer error: {msg.error()}")
                    break

            message_count += 1
            print(f"--- Received message #{message_count} ---")
            try:
                message = json.loads(msg.value().decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON in message #{message_count}: {e}")
                continue

            if message['frame_id'] == -1:
                print("End-of-stream marker received. Exiting.")
                break

            # Decode image
            try:
                frame_bytes = bytes.fromhex(message['data'])
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("cv2.imdecode returned None")
            except Exception as e:
                print(f"[WARNING] Could not decode frame #{message_count}: {e}")
                continue

            try:
                w, h = int(message['width']), int(message['height'])
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid dimensions: width={w}, height={h}")
            except (KeyError, ValueError) as e:
                print(f"[WARNING] Invalid metadata: {e}")
                continue

            # Initialize video writer
            if video_writer is None:
                os.makedirs("detection_output", exist_ok=True)
                input_name = message.get("filename", "output")
                output_path = os.path.join("detection_output", f"{input_name}.avi")
                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Scale and draw region
            try:
                scale_x, scale_y = w / 3840, h / 2160
                scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in region_points]
                scaled_np = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))

                overlay = frame.copy()
                cv2.fillPoly(overlay, [scaled_np], color=(0, 0, 255))  # Red fill
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)   # 20% opacity
                cv2.polylines(frame, [scaled_np], isClosed=True, color=(0, 0, 255), thickness=2)

                print(f"Scaled region drawn for frame #{message_count}")
            except Exception as e:
                print(f"[WARNING] Failed to draw region: {e}")
                continue

            # Apply model and save
            try:
                trackzone.region = scaled_np
                results = trackzone(frame)
                video_writer.write(results.plot_im)
            except Exception as e:
                print(f"[WARNING] Failed to process frame: {e}")
                continue

    finally:
        consumer.close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Kafka consumer stopped. Output video saved.")

if __name__ == "__main__":
    consume_and_process()