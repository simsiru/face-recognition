from utils import insert_delete_embeddings
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert, delete person embeddings in database")
    parser.add_argument(
        "-n",
        "--n_img_person",
        default=1,
        help="Number of embeddings per person"
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

    insert_delete_embeddings(config, args.n_img_person)
