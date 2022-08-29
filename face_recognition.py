import cv2
import numpy as np
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
from pymot.mot import MOT
from utils import DBInterface
import json


# Face recognition
def face_recognition(
    frame, tracking_id, obj_coord, **proc_kwargs
) -> tuple[dict, tuple]:
    """Face recognition function for a MOT tracker"""

    xmin, ymin, xmax, ymax = obj_coord

    # If number of processed objects is more than max limit,
    # delete the first to enter.
    if len(proc_kwargs["proc_obj_info"]) > proc_kwargs["max_n_obj"]:
        if proc_kwargs["last_obj_to_del"] not in proc_kwargs["proc_obj_info"]:
            proc_kwargs["last_obj_to_del"] += 1
        else:
            del proc_kwargs["proc_obj_info"][proc_kwargs["last_obj_to_del"]]

    # If there are no processed objects set the first to enter id
    # as last_obj_to_del.
    if len(proc_kwargs["proc_obj_info"]) == 0:
        proc_kwargs["last_obj_to_del"] = tracking_id

    # If new tracking id, create new entry in the processed objects dict.
    if tracking_id > proc_kwargs["prev_id"]:
        proc_kwargs["prev_id"] = tracking_id
        proc_kwargs["proc_obj_info"][tracking_id] = [1, [], [], False]

    # If the number of performed processings on a object with a certain
    # tracking id is less than needed, do another processing.
    if proc_kwargs["proc_obj_info"][tracking_id][0] <= proc_kwargs["n_det"]:
        bbox_frame = proc_kwargs["mtcnn"](frame[ymin:ymax, xmin:xmax])

        if bbox_frame is not None:
            face_embeddings = proc_kwargs["face_embeddings_model"](
                bbox_frame.unsqueeze(0).to(proc_kwargs["device"])
            )

            dist = np.linalg.norm(
                proc_kwargs["embeddings"]
                - face_embeddings.detach().cpu().numpy(),
                axis=1,
            )
            min_dist_idx = np.argmin(dist)
            min_dist = np.min(dist)

            proc_kwargs["proc_obj_info"][tracking_id][1].append(
                [min_dist_idx, min_dist]
            )

            idx = int(
                proc_kwargs["proc_obj_info"][tracking_id][0]
                * (len(proc_kwargs["proc_animation"]) / proc_kwargs["n_det"])
            )

            proc_kwargs["proc_obj_info"][tracking_id][2] = proc_kwargs[
                "proc_animation"
            ][idx - 1 if idx == len(proc_kwargs["proc_animation"]) else idx]

            proc_kwargs["proc_obj_info"][tracking_id][0] += 1

    # If the number of performed processings on a object with a certain
    # tracking id has sufficed do final processing.
    if (
        proc_kwargs["proc_obj_info"][tracking_id][0]
        == proc_kwargs["n_det"] + 1
    ):
        dist_np_arr = np.array(proc_kwargs["proc_obj_info"][tracking_id][1])

        idx = int(np.mean(dist_np_arr[:, 0]))
        dist = np.mean(dist_np_arr[:, 1])

        if dist < proc_kwargs["min_dist"]:
            proc_kwargs["proc_obj_info"][tracking_id][2] = proc_kwargs[
                "person_names"
            ][idx]

            proc_kwargs["proc_obj_info"][tracking_id][3] = True
        else:
            proc_kwargs["proc_obj_info"][tracking_id][2] = "HOSTILE DETECTED"

        proc_kwargs["proc_obj_info"][tracking_id][0] += 1

    return (
        (
            proc_kwargs["proc_obj_info"],
            proc_kwargs["last_obj_to_del"],
            proc_kwargs["prev_id"],
        ),
        (
            False,
            proc_kwargs["proc_obj_info"][tracking_id][3],
            ((0, 255, 0), (0, 0, 255)),
            proc_kwargs["proc_obj_info"][tracking_id][2],
        ),
    )


def face_recognition_main(args: argparse.Namespace, db_config: dict) -> None:
    
    with open("yolo_data/mscoco_classes.txt") as f:
        classes = f.read().splitlines()

    track_classes = ["person"]

    mot_cfg = {
        "od_classes": classes,
        "od_algo": "yolo",
        "od_wpath": "yolo_data/yolov4-tiny.weights",
        "od_cpath": "yolo_data/yolov4-tiny.cfg",
        "od_nms_thr": 0.4,
        "od_conf_thr": 0.5,
        "od_img_size": 416,
        "od_cuda": True,
        "t_classes": track_classes,
        "t_algo": "deepsort",
        "t_cuda": True,
        "t_metric": "cosine",
        "t_max_cosine_distance": 0.2,
        "t_budget": 100,
        "t_max_iou_distance": 0.7,
        "t_max_age": 70,
        "t_n_init": 3,
    }

    obj_tracking = MOT(mot_cfg)

    if args.video_file_path is not None:
        cap = cv2.VideoCapture(args.video_file_path)
    else:
        cap = cv2.VideoCapture(0)

    prev_id = -1
    n_det = 10
    max_n_obj = 50
    last_obj_to_del = 0
    proc_obj_info = {}
    proc_animation = {
        0: "|",
        1: "|" * 2,
        2: "|" * 3,
        3: "|" * 4,
        4: "|" * 5,
        5: "|" * 6,
        6: "|" * 7,
        7: "|" * 8,
    }

    # Face recognition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    face_embeddings_model = (
        InceptionResnetV1(pretrained="vggface2").eval().to(device)
    )

    mtcnn = MTCNN(device=device)

    min_dist = 0.6

    pgsql_db = DBInterface(db_config)
    sql_script = """
    SELECT person_name, face_embedding
    FROM face_embeddings
    """
    face_emb_df = pgsql_db.execute_sql_script(sql_script, return_result=True)

    face_emb_df["face_embedding"] = face_emb_df["face_embedding"].apply(
        lambda x: pgsql_db.bytes_to_numpy_array(x)
    )
    embeddings = np.concatenate(face_emb_df["face_embedding"].values)

    person_names = face_emb_df["person_name"].values

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

        t1 = time.time()

        (
            t_ojb_info,
            frame_with_bboxes,
            proc_obj_info_tuple,
        ) = obj_tracking.track_objects(
            frame,
            face_recognition,
            prev_id=prev_id,
            n_det=n_det,
            max_n_obj=max_n_obj,
            last_obj_to_del=last_obj_to_del,
            proc_animation=proc_animation,
            proc_obj_info=proc_obj_info,
            device=device,
            face_embeddings_model=face_embeddings_model,
            mtcnn=mtcnn,
            min_dist=min_dist,
            embeddings=embeddings,
            person_names=person_names,
        )

        if None not in proc_obj_info_tuple:
            proc_obj_info, last_obj_to_del, prev_id = proc_obj_info_tuple

        fps = int(1 / (0.0001 + time.time() - t1))

        cv2.putText(
            frame_with_bboxes,
            "FPS: {}".format(fps),
            (0, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("Face recognition", frame_with_bboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition")
    parser.add_argument(
        "-v",
        "--video_file_path",
        default=None,
        help="Use a video for tracking and if the path is not provided use a webcam",
    )
    parser.add_argument(
        "-c",
        "--db_config",
        default='db_config.json',
        help="Path to database config"
    )
    args = parser.parse_args()

    with open(args.db_config, 'r') as f:
        config = json.load(f)

    face_recognition_main(args, config)
