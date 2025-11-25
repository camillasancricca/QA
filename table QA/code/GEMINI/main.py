import pandas as pd
import time
import json
import re
import numpy as np
import string
import random
import warnings
import google
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from scipy.spatial.distance import cosine
from collections import defaultdict
from google import genai
nltk.download('punkt')
import re
################################################################ DIRTY FUNCTIONS ############################################################################
# percentuali
perc = [0.10,0.30,0.50]

#COMPLETENESS
def dirty_completeness(seed, df):
    np.random.seed(seed)
    df_pandas = df.copy()
    df_list = []
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p,1-p]
        for col in df_dirt.columns:
            rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)
            df_dirt.loc[rand == True,col]=np.nan
        df_list.append(df_dirt)
    return df_list


# SEMANTIC_ACCURACY
def typo(message):
    message = list(message)
    typo_prob = 0.2
    if len(message) <= 1:
        return ''.join(message)
    n_chars_to_flip = max(round(len(message) * typo_prob),1)
    capitalization = [False] * len(message)
    for i in range(len(message)):
        capitalization[i] = message[i].isupper()
        message[i] = message[i].lower()
    pos_to_flip = []
    for i in range(n_chars_to_flip):
        pos_to_flip.append(random.randint(0, len(message) - 1))
    nearbykeys = {
        'a': ['q','w','s','x','z'],
        'b': ['v','g','h','n'],
        'c': ['x','d','f','v'],
        'd': ['s','e','r','f','c','x'],
        'e': ['w','s','d','r'],
        'f': ['d','r','t','g','v','c'],
        'g': ['f','t','y','h','b','v'],
        'h': ['g','y','u','j','n','b'],
        'i': ['u','j','k','o'],
        'j': ['h','u','i','k','n','m'],
        'k': ['j','i','o','l','m'],
        'l': ['k','o','p'],
        'm': ['n','j','k','l'],
        'n': ['b','h','j','m'],
        'o': ['i','k','l','p'],
        'p': ['o','l'],
        'q': ['w','a','s'],
        'r': ['e','d','f','t'],
        's': ['w','e','d','x','z','a'],
        't': ['r','f','g','y'],
        'u': ['y','h','j','i'],
        'v': ['c','f','g','v','b'],
        'w': ['q','a','s','e'],
        'x': ['z','s','d','c'],
        'y': ['t','g','h','u'],
        'z': ['a','s','x'],
        ' ': ['c','v','b','n','m'],
        '1': ['q'],
        '2': ['q','w'],
        '3': ['w','e'],
        '4': ['e','r'],
        '5': ['r','t'],
        '6': ['t','y'],
        '7': ['y','u'],
        '8': ['u','i'],
        '9': ['i','o'],
        '0': ['o','p'],
    }
    for pos in pos_to_flip:
        try:
            typo_arrays = nearbykeys[message[pos]]
            message[pos] = np.random.choice(typo_arrays)
        except:
            break
    for i in range(len(message)):
        if (capitalization[i]):
            message[i] = message[i].upper()
    message = ''.join(message)
    return message


def insert_typos(seed, df):
    np.random.seed(seed)
    df_pandas = df.copy()
    df_list = []

    for p in perc:
        df_dirty = df_pandas.copy()
        acc = [p, 1 - p]

        for col in df_dirty.columns:
            if pd.api.types.is_object_dtype(df_dirty[col]) or df_dirty[col].dtype == "category":
                df_dirty[col] = df_dirty[col].astype(str)
                rand = np.random.choice([True, False], size=df_dirty.shape[0], p=acc)
                selected = df_dirty.loc[rand, col].copy()
                for i in selected.index:
                    selected.loc[i] = typo(str(selected.loc[i]))
                df_dirty.loc[rand, col] = selected

        df_list.append(df_dirty)
    return df_list

#OUTLIERS
def out_of_range(minimum, maximum):
    dist = maximum - minimum
    if dist == 0:
        dist = 1
    if random.random() < 0.5:
        return random.uniform(maximum, maximum + dist)
    else:
        return random.uniform(minimum - dist, minimum)

def is_convertible_number(x):
    try:
        float(x)
        return True
    except:
        return False

def insert_outliers(seed, df):
    np.random.seed(seed)
    random.seed(seed)
    df_pandas = df.copy()
    df_list = []

    for p in perc:
        df_dirty = df_pandas.copy()
        acc = [p, 1 - p]

        for col in df_dirty.columns:
            numeric_mask = df_dirty[col].apply(is_convertible_number)

            if numeric_mask.sum() == 0:
                continue

            col_numeric = df_dirty.loc[numeric_mask, col].apply(float)
            minimum = float(col_numeric.min())
            maximum = float(col_numeric.max())

            rand = np.random.choice([True, False], size=col_numeric.shape[0], p=acc)
            selected_indices = col_numeric[rand].index

            for i in selected_indices:
                try:
                    df_dirty.at[i, col] = out_of_range(minimum, maximum)
                except:
                    continue
        df_list.append(df_dirty)

    return df_list




#DUPLICATION
def insert_duplicates(seed, df):
    np.random.seed(seed)
    results = []
    for p in perc:
        n_duplicates = int(len(df) * p)
        n_normal = n_duplicates // 2
        n_typos = n_duplicates // 4
        n_missing = n_duplicates - n_normal - n_typos  # Totale corretto anche se dispari
        normal_rows = df.sample(n=n_normal, replace=True).copy()
        typo_rows = df.sample(n=n_typos, replace=True).copy()
        missing_rows = df.sample(n=n_missing, replace=True).copy()
        for idx, row in typo_rows.iterrows():
            n_cells_to_modify = max(1, int(len(row) * 0.1))
            cols_to_modify = np.random.choice(df.columns, n_cells_to_modify, replace=False)
            for col in cols_to_modify:
                original = str(row[col])
                typo_value = typo(original)
                row[col] = typo_value
        for idx, row in missing_rows.iterrows():
            n_cells_to_null = max(1, int(len(row) * 0.1))
            cols_to_null = np.random.choice(df.columns, n_cells_to_null, replace=False)
            for col in cols_to_null:
                row[col] = ""
        df_with_dups = pd.concat([df, normal_rows, typo_rows, missing_rows], ignore_index=True)
        results.append(df_with_dups)

    return results

#SYNTACTIC_ACCURCAY
def syntactic_accuracy(seed, df):
    np.random.seed(seed)

    df_pandas = df.copy()
    df_list = []

    for p in perc:
        df_dirt = df_pandas.copy()
        n_cols_to_shuffle = max(1, int(len(df_dirt.columns) * p))  # almeno 1 colonna
        cols_to_shuffle = np.random.choice(df_dirt.columns, size=n_cols_to_shuffle, replace=False)

        for col in cols_to_shuffle:
            shuffled_values = df_dirt[col].sample(frac=1, random_state=seed).reset_index(drop=True)
            df_dirt[col] = shuffled_values

        df_list.append(df_dirt)
    return df_list
############################################################## ACCESSORIES FUNCTIONS ###################################################################
# Function to convert JSON data into CSV format
def json_to_csv(json_data):
    headers = [col['text'] for col in json_data['columns']]  # Extract headers
    rows = []
    for row in json_data['rows']:
        cells = [cell['text'] for cell in row['cells']]
        rows.append(cells)

    return headers, rows

# Function to load the questions from a text file
def load_questions(file_path):
    with open(file_path, 'r') as f:
        questions = [line.strip() for line in f.readlines()]
    return questions

# Function to save the answers in a text file
def save_answers(answer, output_file):
    with open(output_file, 'a') as f:
        f.write(answer + '\n')

def df_to_csv_string(df):
    original_headers = [
        "" if col.startswith("Unnamed") else col.split(".")[0]
        for col in df.columns
    ]
    rows = df.astype(str).values.tolist()
    return "\n".join([",".join(original_headers)] + [",".join(row) for row in rows])

def df_creation(line):
    original = json.loads(line)
    columns = [col["text"] for col in original["columns"]]
    rows = [[cell["text"] for cell in row["cells"]] for row in original["rows"]]
    df = pd.DataFrame(rows, columns=columns)
    df.columns = [f'Unnamed_{i}' if col == '' else col for i, col in enumerate(df.columns)]
    df.columns = make_unique_columns(df.columns)
    return df

def make_unique_columns(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col == '':
            col = 'Unnamed'
        count = seen.get(col, 0)
        if count == 0:
            new_columns.append(col)
        else:
            new_columns.append(f"{col}.{count}")
        seen[col] = count + 1
    return new_columns

def normalize_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    return text

# Raggruppamento per tabelle equivalenti
def normalize_table(table):
    columns = tuple(col['text'].strip().lower() for col in table['columns'])
    rows = tuple(tuple(cell['text'].strip().lower() for cell in row['cells']) for row in table['rows'])
    return (columns, rows)


################################################################### REQUEST FUNCTIONS ############################################################################
client = genai.Client(api_key="AIzaSyBKoG1ZXtcvNhyi-GUORAXxRDlwkQGQ8b4")
def query_table_multiple(table_csv, questions):
    prompt = f"""You are an expert at reading CSV tables.

    Here is a table in CSV format:
    {table_csv}
    Answer the following questions using only the information in the table. 
    Give each answer on a new line with the corrisponding number, in order. Do not add explanations or repeat the question.

    Questions:
    """ + "\n".join([f"{i + 1}. {q}?" for i, q in enumerate(questions)])

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    output = response.text
    answers = [line.strip() for line in output.split('\n') if line.strip()]
    time.sleep(10)
    return answers
############################################################# SIMILARITY FUNCTIONS #######################################################################
# Similarità TF-IDF
def tfidf_similarity(s1, s2):
    try:
        tfidf = TfidfVectorizer().fit([s1, s2])
        tfidf_matrix = tfidf.transform([s1, s2])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except ValueError:
        return 0.0

# Cross-Encoder
def cross_encoder_similarity(answer1, answer2):
    score = cross_encoder_model.predict([(answer1, answer2)])[0]
    return float(score)

# Calcolo metriche
def compute_similarities(answer1, answer2):
    sims = {}
    sims['TF-IDF Similarity'] = tfidf_similarity(answer1, answer2)
    vecs_msmarco = msmarco_model.encode([answer1, answer2])
    sims['MSMarco Cosine Similarity'] = cosine_similarity_vectors(vecs_msmarco[0], vecs_msmarco[1])
    sims['Cross-Encoder Similarity'] = cross_encoder_similarity(answer1, answer2)
    return sims

# Coseno tra vettori
def cosine_similarity_vectors(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return 1 - cosine(vec1, vec2)

# Decisione finale
def classify_answer(sims):
    tfidf = sims['TF-IDF Similarity']
    msmarco = sims['MSMarco Cosine Similarity']
    ces = sims['Cross-Encoder Similarity']

    if tfidf >= 0.82:
        return "RIGHT"
    elif msmarco > 0.7 and ces > 0.671:
        return "RIGHT"
    else:
        return "WRONG"


#################################################################### MAIN ################################################################################


def clean_answer(ans):
    return re.sub(r"^\d+\.\s*", "", ans).strip()

# FUNZIONE CENTRALE

def evaluate_table_variants(table_csv, question_list, r_answer_list, df_variants, variant_name, acc_stats):
    save = f"\n{variant_name.upper()}\n"
    for i, df in enumerate(df_variants):
        csv_version = df_to_csv_string(df)
        gen_answers = query_table_multiple(csv_version, [question_list[0]])

        for idx, (gen_ans, gt_ans) in enumerate(zip(gen_answers, r_answer_list)):
            cleaned_gen_ans = clean_answer(gen_ans)
            sims = compute_similarities(gt_ans, cleaned_gen_ans)
            result = classify_answer(sims)

            perc_str = f"{int((1 - perc[i]) * 100)}%"
            save += f"{perc_str} - Q{idx+1}: {gen_ans} => {result}\n"
            for metric, value in sims.items():
                save += f"  {metric}: {value:.4f}\n"

            acc_stats[variant_name][i]["total"] += 1
            if result == 'RIGHT':
                acc_stats[variant_name][i]["correct"] += 1

            save += f"> Accuracy {variant_name} {perc_str} Q{idx+1}: {acc_stats[variant_name][i]['correct']}/{acc_stats[variant_name][i]['total']}\n"
    return save

msmarco_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
cross_encoder_model = CrossEncoder('cross-encoder/stsb-roberta-base')

# Load the questions from the text file
questions = load_questions('questions.txt')
right_answers = load_questions('r_answers.txt')

# Verifica la lunghezza di ciascun file/lista
print(f"Lunghezza questions: {len(questions)}")
print(f"Lunghezza right_answers: {len(right_answers)}")
accuracy_stats = {
    "clean": {"correct": 0, "total": 0},
    "completeness": [{"correct": 0, "total": 0} for _ in perc],
    "typos_accuracy": [{"correct": 0, "total": 0} for _ in perc],
    "outliers_accuracy": [{"correct": 0, "total": 0} for _ in perc],
    "duplication": [{"correct": 0, "total": 0} for _ in perc],
    "syntactic_accuracy": [{"correct": 0, "total": 0} for _ in perc],
}

# === PRECARICAMENTO DATI ===
with open("tables.jsonl", "r", encoding="utf-8") as tf, open("questions.txt", "r") as qf, open("r_answers.txt", "r") as af:
    tables_raw = [json.loads(line.strip()) for line in tf]
    questions = [line.strip() for line in qf]
    right_answers = [line.strip() for line in af]

assert len(tables_raw) == len(questions) == len(right_answers)

# Raggruppa per tabella normalizzata
normalized_map = defaultdict(list)
for idx, table in enumerate(tables_raw):
    key = normalize_table(table)
    normalized_map[key].append(idx)

# === ESECUZIONE SU OGNI GRUPPO ===
for group_id, indexes in enumerate(normalized_map.values()):

    sample_table = tables_raw[indexes[0]]
    headers, rows = json_to_csv(sample_table)
    table_csv = "\n".join([" ,".join(row) for row in [headers] + rows])
    n_df = df_creation(json.dumps(sample_table))

    question_list = [questions[i] for i in indexes]
    r_answer_list = [right_answers[i] for i in indexes]

    if(group_id==1):
        save = f"\n### TABLE GROUP {group_id+1}\n"
        save += f"Questions:\n" + "\n".join(question_list) + "\n"
        save += f"Right Answers:\n" + "\n".join(r_answer_list) + "\n"

        # CLEAN
        clean_answers = query_table_multiple(table_csv, [question_list[0]])
        for i, (gen_ans, gt_ans) in enumerate(zip(clean_answers, r_answer_list)):
            cleaned_gen_ans = clean_answer(gen_ans)
            sims = compute_similarities(gt_ans, cleaned_gen_ans)
            result = classify_answer(sims)
            save += f"Clean - Q{i+1}: {gen_ans} => {result}\n"
            for metric, value in sims.items():
                save += f"  {metric}: {value:.4f}\n"
            accuracy_stats["clean"]["total"] += 1
            if result == 'RIGHT':
                accuracy_stats["clean"]["correct"] += 1
            save += f"> Accuracy Clean Q{i+1}: {accuracy_stats['clean']['correct']}/{accuracy_stats['clean']['total']}\n"

        for variant_name, dirty_func in [
            ("completeness", dirty_completeness),
            ("typos_accuracy", insert_typos),
            ("outliers_accuracy", insert_outliers),
            ("duplication", insert_duplicates),
            ("syntactic_accuracy", syntactic_accuracy)
        ]:
            for seed_variant in range(3):
                dirty_versions = dirty_func(seed=seed_variant, df=n_df)
                save += f"\n### {variant_name.upper()} with SEED {seed_variant}\n"
                save += evaluate_table_variants(table_csv, question_list[0], r_answer_list, dirty_versions, variant_name, accuracy_stats)

        print(save)
        save_answers(save, "answers.txt")

## interruzioni dopo tabelle: 2-8-29-34-39-49-70-86-97
## interruzioni dopo tabelle: 7-11-15-19-23-27-31-35-39-40-42-46-49-50
## interruzioni dopo tabelle: 3-7-11-15-20-24-28-32-33-37-41-42-46-50
## interruzioni dopo tabelle: 4-5-6-8-12-13-14-15-18-22-23-25-29-33-37-41-45-49-50
## interruzioni dopo tabelle: 2-6-10-13-16-19-27-31-35-39-43-47-51-55-59-65-68-69-72-77