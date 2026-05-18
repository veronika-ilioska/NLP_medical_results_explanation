import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_SCHEMA = os.getenv("DB_SCHEMA")

DEFAULT_OUTPUT_PATH = Path("data/lab_summaries_export.csv")


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def validate_config():
    missing = [
        name
        for name, value in {
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_SCHEMA": DB_SCHEMA,
        }.items()
        if not value
    ]

    if missing:
        joined = ", ".join(missing)
        sys.exit(f"Missing required environment variables: {joined}")


def build_export_query():
    return sql.SQL(
        """
        SELECT
            summary_id,
            subject_id,
            hadm_id,
            charttime,
            generated_text,
            prompt,
            model_used,
            created_at
        FROM {schema}.lab_summaries s
        ORDER BY
            subject_id,
            charttime,
            summary_id
        """
    ).format(schema=sql.Identifier(DB_SCHEMA))


def export_csv(output_path):
    with get_connection() as conn:
        query = build_export_query().as_string(conn)
        df = pd.read_sql_query(query, conn)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Exported {len(df)} rows to {output_path}")


def main():
    validate_config()

    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    export_csv(output_path)


if __name__ == "__main__":
    main()
