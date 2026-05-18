import os
import sys
import logging
import psycopg2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_SCHEMA   = os.getenv("DB_SCHEMA")

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL")

PATIENT_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# exclude oxygen machine paramteres
EXCLUDED_LABELS = (
    'PEEP', 'Tidal Volume', 'Oxygen',
    'Required O2', 'O2 Flow', 'Temperature', 'WBC Count'
)

# must not be zero
IMPOSSIBLE_ZERO_LABELS = (
    'Creatinine', 'Hemoglobin', 'Hematocrit',
    'Red Blood Cells', 'MCV', 'MCH', 'MCHC',
    'Platelet Count', 'White Blood Cells'
)


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )


def get_llm_client():
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def build_query():
    excluded         = ", ".join(f"'{l}'" for l in EXCLUDED_LABELS)
    impossible_zeros = ", ".join(f"'{l}'" for l in IMPOSSIBLE_ZERO_LABELS)

    return f"""
    WITH panel_stats AS (
        SELECT
            l.subject_id,
            l.hadm_id,
            l.charttime,
            COUNT(DISTINCT d.label)    AS unique_tests,
            COUNT(DISTINCT d.category) AS categories,
            COUNT(DISTINCT CASE WHEN l.flag = 'abnormal'
                THEN d.label END)      AS abnormal_tests
        FROM {DB_SCHEMA}.labevents l
        JOIN {DB_SCHEMA}.d_labitems d ON l.itemid = d.itemid
        WHERE d.fluid = 'Blood'
          AND l.valuenum IS NOT NULL
          AND l.valuenum > 0
          AND d.label NOT IN ({excluded})
          AND NOT (
              l.valuenum = 0
              AND d.label IN ({impossible_zeros})
          )
        GROUP BY l.subject_id, l.hadm_id, l.charttime
    ),
    best_panel_per_patient AS (
        SELECT DISTINCT ON (subject_id)
            subject_id, hadm_id, charttime
        FROM panel_stats
        ORDER BY subject_id, abnormal_tests DESC
    ),
    selected_patients AS (
        SELECT * FROM best_panel_per_patient
        LIMIT {PATIENT_LIMIT}
    )
    SELECT DISTINCT
        l.subject_id,
        l.hadm_id,
        l.charttime,
        d.label,
        d.category,
        d.loinc_code,
        l.valuenum,
        l.valueuom,
        CASE
            WHEN l.flag IN ('abnormal', 'delta') THEN 'abnormal'
            WHEN l.flag IS NULL THEN 'normal'
            ELSE 'normal'
        END AS flag_clean
    FROM {DB_SCHEMA}.labevents l
    JOIN {DB_SCHEMA}.d_labitems d ON l.itemid = d.itemid
    JOIN selected_patients sp
        ON  l.subject_id = sp.subject_id
        AND l.charttime  = sp.charttime
    WHERE d.fluid = 'Blood'
      AND l.valuenum IS NOT NULL
      AND l.valuenum > 0
      AND d.label NOT IN ({excluded})
      AND NOT (
          l.valuenum = 0
          AND d.label IN ({impossible_zeros})
      )
    ORDER BY l.subject_id, d.category, d.label
    """


def build_prompt(panel_df):
    lines = []
    for _, row in panel_df.iterrows():
        unit = row['valueuom'] if pd.notna(row['valueuom']) else 'no unit'
        lines.append(
            f"- {row['label']}: {row['valuenum']} {unit} [{row['flag_clean']}]"
        )
    tests_text = "\n".join(lines)

    return f"""You are writing patient-friendly explanations of laboratory test results.

Task:
For each blood test result, write exactly one very short sentence explaining what this result may suggest in the body.

Return the answer in exactly this format:

- Test Name: value unit - one short explanation.
- Test Name: value unit - one short explanation.

General Overview: one short paragraph summarizing the overall pattern.

Strict rules:
- Use one bullet line per test.
- Keep the tests in the same order as the input.
- Each bullet must follow exactly this pattern:
  - Test Name: value unit - Explanation.
- Include the test name, value, and unit exactly as given.
- Write only one sentence after the dash.
- Keep each explanation under 18 words.
- Use simple language for a non-medical reader.
- Use cautious wording such as "may suggest", "can suggest", "may reflect", or "appears".
- If the result is normal, say what body function appears generally within the expected range.
- If the result is abnormal, explain the possible body system involved.
- Do not diagnose diseases.
- Do not recommend treatment.
- Do not say the body "is" damaged, failing, or diseased.
- End with exactly one paragraph starting with:
  General Overview:
- Do not add any other headers, numbering, markdown tables, or extra text.

Example output style:
- Hemoglobin: 10.5 g/dL - May reflect a lower amount of oxygen-carrying protein in the blood.
- White Blood Cells: 12.0 K/uL - Can suggest an immune response, such as infection or inflammation.

General Overview: The results show ...

BLOOD TEST RESULTS:

{tests_text}"""


def call_llm(client, prompt):
    log.info(f"Prompt sent to LLM:\n{prompt}")
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce concise patient-friendly lab explanations and "
                    "must follow the requested output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        top_p=1,
    )
    return response.choices[0].message.content


def store_result(cursor, subject_id, hadm_id, charttime,
                 prompt, generated_text, panel_df):

    cursor.execute(f"""
        INSERT INTO {DB_SCHEMA}.lab_summaries
            (subject_id, hadm_id, charttime, prompt, generated_text, model_used)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (subject_id, charttime) DO UPDATE
            SET generated_text = EXCLUDED.generated_text,
                prompt         = EXCLUDED.prompt,
                model_used     = EXCLUDED.model_used,
                created_at     = NOW()
        RETURNING summary_id
    """, (
        int(subject_id),
        int(hadm_id) if pd.notna(hadm_id) else None,
        charttime,
        prompt,
        generated_text,
        NVIDIA_MODEL
    ))

    summary_id = cursor.fetchone()[0]

    cursor.execute(f"""
        DELETE FROM {DB_SCHEMA}.lab_summary_items
        WHERE summary_id = %s
    """, (summary_id,))

    for _, row in panel_df.iterrows():
        cursor.execute(f"""
            INSERT INTO {DB_SCHEMA}.lab_summary_items
                (summary_id, label, category, loinc_code,
                 valuenum, valueuom, flag_clean)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            summary_id,
            row['label'],
            row['category'],
            row.get('loinc_code'),
            row['valuenum'],
            row['valueuom'] if pd.notna(row['valueuom']) else None,
            row['flag_clean']
        ))

    return summary_id


def main():
    log.info(f"Starting pipeline | PATIENT_LIMIT: {PATIENT_LIMIT}")

    conn   = get_connection()
    client = get_llm_client()

    log.info("Fetching panels from DB...")
    df = pd.read_sql(build_query(), conn)

    panels = list(df.groupby(['subject_id', 'charttime']))
    log.info(f"Fetched {len(df)} rows across {len(panels)} panels")

    cursor = conn.cursor()

    for i, ((subject_id, charttime), panel) in enumerate(panels, start=1):
        hadm_id        = panel['hadm_id'].iloc[0]
        abnormal_count = (panel['flag_clean'] == 'abnormal').sum()
        normal_count   = (panel['flag_clean'] == 'normal').sum()

        log.info(f"--- Panel {i}/{len(panels)} ---")
        log.info(f"subject_id: {subject_id} | hadm_id: {hadm_id} | charttime: {charttime}")
        log.info(f"Tests: {len(panel)} total | {abnormal_count} abnormal | {normal_count} normal")
        log.info("Aggregated tests:\n" + panel[
            ['label', 'category', 'valuenum', 'valueuom', 'flag_clean']
        ].to_string(index=False))

        prompt         = build_prompt(panel)
        generated_text = call_llm(client, prompt)

        log.info(f"LLM response:\n{generated_text}")

        summary_id = store_result(
            cursor, subject_id, hadm_id, charttime,
            prompt, generated_text, panel
        )
        conn.commit()
        log.info(f"Stored summary_id: {summary_id} ✓")

    cursor.close()
    conn.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
