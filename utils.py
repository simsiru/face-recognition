import cv2
import numpy as np
import psycopg2 as pg
import io
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


def draw_bbox_tracking(
    coords: tuple,
    height: int,
    frame: np.ndarray,
    tracking_id: int,
    class_name: str="",
    rand_colors: bool=True,
    rec_bool: bool=False,
    colors: tuple=((0, 255, 0), (0, 0, 255)),
    info_text: str="",
    unknown_obj_info: str="UNKNOWN",
) -> None:
    """Function for visualizing tracking bounding boxes"""

    assert len(colors) == 2, "Colors must a list of tuples of length 2"

    xmin, ymin, xmax, ymax = coords

    if rand_colors:
        np.random.seed(tracking_id)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        color = (int(r * 255), int(g * 255), int(b * 255))
    else:
        if rec_bool:
            color = colors[0]
        else:
            info_text = unknown_obj_info
            color = colors[1]

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    text = "{} ID: {} [{}]".format(class_name, tracking_id, info_text)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, 0.75, thickness=1
    )

    if (ymax + text_height) > height:
        cv2.rectangle(
            frame,
            (xmin, ymax),
            (xmin + text_width, ymax - text_height - baseline),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax - 4),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.rectangle(
            frame,
            (xmin, ymax + text_height + baseline),
            (xmin + text_width, ymax),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax + text_height + 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )


class DBInterface:
    """Class for interface with PostgreSQL database"""

    def __init__(self, db_config: dict):
        self.host = db_config['hostname']
        self.dbname = db_config['database']
        self.username = db_config['username']
        self.password = db_config['password']
        self.port = db_config['port_id']

    def execute_sql_script(
        self, sql_script, values_insert=None, return_result=False
    ):
        conn = None
        cur = None
        df = None

        try:
            with pg.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.username,
                password=self.password,
                port=self.port,
            ) as conn:

                if return_result:
                    df = pd.read_sql_query(sql_script, conn)
                else:
                    with conn.cursor() as cur:
                        if values_insert is not None:
                            cur.execute(sql_script, values_insert)
                        else:
                            cur.execute(sql_script)

        except Exception as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

        if return_result:
            return df

    def numpy_array_to_bytes(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return out.read()

    def bytes_to_numpy_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)


def insert_delete_embeddings(db_config: dict, n_img_class: int=1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    mtcnn = MTCNN(device=device)

    curr_img_n = 0

    person_name = ""

    name_saving_mode = True
    emb_saving_mode = False

    db_interface = DBInterface(db_config)

    cap = cv2.VideoCapture(0)

    while True:
        ret, rgb = cap.read()
        if not ret:
            break

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

        rgb_with_bboxes = rgb.copy()

        face = mtcnn(rgb)

        if face is not None:
            tmp = face.numpy().copy().transpose(1, 2, 0)
            tmp = cv2.normalize(
                tmp,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            tmp = tmp.astype(np.uint8)
            rgb_with_bboxes[0 : tmp.shape[0], 0 : tmp.shape[1]] = tmp

        cv2.imshow("Webcam", rgb_with_bboxes)

        if cv2.waitKey(25) & 0xFF == ord("a") and name_saving_mode:
            person_name = input("Enter next person's name: ")
            print(f"Currently collecting {person_name} data")
            curr_img_n = 0

            name_saving_mode = False
            emb_saving_mode = True

        if (
            cv2.waitKey(25) & 0xFF == ord("s")
            and emb_saving_mode
            and curr_img_n <= n_img_class - 1
        ):
            if face is None:
                continue

            embeddings = emb_model(face.unsqueeze(0).to(device)).detach().cpu()

            if curr_img_n == 0:
                face_embeddings = embeddings

            else:
                face_embeddings = torch.cat((face_embeddings, embeddings), 0)

            if curr_img_n == n_img_class - 1:
                if n_img_class > 1:
                    face_embeddings = (
                        face_embeddings.mean(dim=0).unsqueeze(0).numpy()
                    )
                else:
                    face_embeddings = face_embeddings.numpy()

                sql_script = """
                INSERT INTO face_embeddings(person_name, face_embedding, timestamp)
                VALUES (%s, %s, current_timestamp)
                """
                db_interface.execute_sql_script(
                    sql_script,
                    (
                        person_name,
                        db_interface.numpy_array_to_bytes(face_embeddings),
                    ),
                )

                name_saving_mode = True
                emb_saving_mode = False

            curr_img_n += 1

            print(f"Embedding {curr_img_n} saved")

            sql_script = """
            SELECT *
            FROM face_embeddings
            """
            print(
                db_interface.execute_sql_script(sql_script, return_result=True)
            )

        if cv2.waitKey(25) & 0xFF == ord("d") and name_saving_mode:
            delete_person_name = input("Enter a person's name to delete: ")

            sql_script = """
            DELETE FROM face_embeddings
            WHERE person_name = %s
            """
            db_interface.execute_sql_script(sql_script, (delete_person_name,))

            print(f"{delete_person_name} data deleted")

            sql_script = """
            SELECT *
            FROM face_embeddings
            """
            print(
                db_interface.execute_sql_script(sql_script, return_result=True)
            )

    sql_script = """
    SELECT *
    FROM face_embeddings
    """
    print(db_interface.execute_sql_script(sql_script, return_result=True))
