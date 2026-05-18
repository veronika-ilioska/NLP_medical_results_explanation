import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_SCHEMA   = os.getenv("DB_SCHEMA")


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )


def create_tables(cursor):
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.lab_summaries (
            summary_id     SERIAL PRIMARY KEY,
            subject_id     INT NOT NULL,
            hadm_id        INT,
            charttime      TIMESTAMP NOT NULL,
            generated_text TEXT,
            prompt         TEXT,
            model_used     VARCHAR(100),
            created_at     TIMESTAMP DEFAULT NOW(),
            UNIQUE (subject_id, charttime)
        )
    """)
    print(f"Table {DB_SCHEMA}.lab_summaries created")

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.lab_summary_items (
            item_id    SERIAL PRIMARY KEY,
            summary_id INT NOT NULL
                REFERENCES {DB_SCHEMA}.lab_summaries(summary_id),
            label      VARCHAR(200),
            category   VARCHAR(100),
            loinc_code VARCHAR(50),
            valuenum   DOUBLE PRECISION,
            valueuom   VARCHAR(20),
            flag_clean VARCHAR(20)
        )
    """)
    print(f"Table {DB_SCHEMA}.lab_summary_items created")


def main():
    conn   = get_connection()
    cursor = conn.cursor()

    create_tables(cursor)
    conn.commit()

    cursor.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()