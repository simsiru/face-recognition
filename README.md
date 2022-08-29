# Face recognition

> Perform face recognition on video file or webcam

## Installation

To use the script these packages must be installed:

- torch
- torchvision
- opencv
- facenet_pytorch
- pymot
- psycopg2

Note: It is recommended to use opencv compiled with CUDA support, because algorithms that use opencv's darknet
backend like YOLOv3/v4 run an order of magnitude faster on a GPU.

## Usage

Before running any of the scripts the postgreSQL database must have the face_embeddings table which can be created
with the following query:

```sql
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    person_name VARCHAR(32) NOT NULL,
    face_embedding BYTEA NOT NULL,
    timestamp timestamp NOT NULL
)
```

Run the face recognition script with the following arguments:

- --video_file_path or -v - use a video for tracking and if the path is not provided use a webcam (default is webcam)
- --db_config or -c - configuration json file for postgreSQL database (default is db_config.json)

```sh
python3 face_recognition.py -v PATH_TO_VIDEO_FILE -c PATH_TO_CONFIG
```

To add face embeddings to a database use the add_face_embeddings.py script with arguments:

- --n_img_person or -n - number of embeddings per person (default is 1)
- --db_config or -c - configuration json file for postgreSQL database (default is db_config.json)

```sh
python3 add_face_embeddings.py -n 1 -c PATH_TO_CONFIG
```

## License

[MIT](https://choosealicense.com/licenses/mit/)