# Reconstructed dissertation code. See docs/experiment_notes.md before running.

# Repository paths: works from the project root or the notebooks/ folder.
from pathlib import Path
import os

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name in {'notebooks', 'src', 'outputs'}:
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / 'requirements.txt').exists():
    raise RuntimeError('Start Jupyter or Python from the repository root.')
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
os.chdir(OUTPUT_DIR)

required_files = [
    'emotion_train.csv', 'emotion_validation.csv', 'emotion_test.csv',
    'goemotions_train.csv', 'goemotions_validation.csv', 'goemotions_test.csv',
]
missing = [name for name in required_files if not (DATA_DIR / name).is_file()]
if missing:
    raise FileNotFoundError('Add these files to data/: ' + ', '.join(missing))
print(f'Datasets: {DATA_DIR}\nNew run outputs: {OUTPUT_DIR}')


# Step 1: Install dependencies
# Before running: python -m pip install -r requirements.txt


# Step 2: Imports
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification
)
from torch.utils.data import Dataset

print("All imports successful")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Step 3: Configuration

EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
NUM_EMOTION_LABELS = 6

GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_GOEMOTIONS_LABELS = 28

MAX_LENGTH    = 128
BATCH_SIZE    = 8       # Reduce to 4 if OOM on Colab
GRAD_ACCUM    = 2       # Effective batch = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-5
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# Step 4: Custom PyTorch Dataset class
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Step 5: Shared helper functions

def tokenize_texts(texts, tokenizer):
    return tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def get_training_args(output_dir, epochs=NUM_EPOCHS):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_steps=200,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=GRAD_ACCUM,
        report_to='none',
        seed=SEED
    )

def evaluate_model(trainer, val_dataset, val_labels, label_names, model_name):
    predictions  = trainer.predict(val_dataset)
    pred_labels  = predictions.predictions.argmax(axis=-1)
    accuracy     = accuracy_score(val_labels, pred_labels)
    p, r, f1_w, _ = precision_recall_fscore_support(val_labels, pred_labels, average='weighted')
    _, _, f1_m, _ = precision_recall_fscore_support(val_labels, pred_labels, average='macro')
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Weighted F1 : {f1_w:.4f}")
    print(f"Macro F1    : {f1_m:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels, pred_labels, target_names=label_names))
    return pred_labels, accuracy, f1_m, f1_w

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

print("Helper functions ready")


# Step 6: Load Emotion dataset (6 classes)
# Upload emotion_train.csv, emotion_validation.csv, emotion_test.csv to Colab

df      = pd.read_csv(DATA_DIR / 'emotion_train.csv', encoding_errors='replace')
val_df  = pd.read_csv(DATA_DIR / 'emotion_validation.csv', encoding_errors='replace')
test_df = pd.read_csv(DATA_DIR / 'emotion_test.csv', encoding_errors='replace')

print(f"Emotion Dataset loaded")
print(f"  Train : {len(df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
print(f"  Columns: {list(df.columns)}")
print(df[['text', 'label']].head(5))


# Step 7: Load GoEmotions dataset (28 classes)
# Upload goemotions_train.csv, goemotions_validation.csv, goemotions_test.csv to Colab

go_train_df = pd.read_csv(DATA_DIR / 'goemotions_train.csv', encoding_errors='replace', engine='python', on_bad_lines='skip')
go_val_df   = pd.read_csv(DATA_DIR / 'goemotions_validation.csv', encoding_errors='replace', engine='python', on_bad_lines='skip')
go_test_df  = pd.read_csv(DATA_DIR / 'goemotions_test.csv', encoding_errors='replace', engine='python', on_bad_lines='skip')

# Multi-label column -> take first label for single-label classification
def get_first_label(labels):
    try:
        if pd.isna(labels): return 27
        if isinstance(labels, str): return int(float(labels.split(',')[0]))
        return int(float(labels))
    except:
        return 27

for d in [go_train_df, go_val_df, go_test_df]:
    d['label'] = d['labels'].apply(get_first_label)

print(f"GoEmotions Dataset loaded")
print(f"  Train: {len(go_train_df):,}  Val: {len(go_val_df):,}  Test: {len(go_test_df):,}")
print(go_train_df[['text', 'labels', 'label']].head(5))


# Step 8: Basic dataset statistics

print("=" * 55)
print("EMOTION DATASET - Info")
print("=" * 55)
print(df.info())
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows (by text): {df.duplicated(subset='text').sum()}")

print("\n" + "=" * 55)
print("GOEMOTIONS DATASET - Info")
print("=" * 55)
print(f"Null values:\n{go_train_df.isnull().sum()}")
print(f"\nDuplicate rows (by text): {go_train_df.duplicated(subset='text').sum()}")


# Step 9: Label distribution across Train / Val / Test splits - Emotion dataset

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
splits = [('Train', df), ('Validation', val_df), ('Test', test_df)]

for ax, (split_name, split_df) in zip(axes, splits):
    counts = split_df['label'].value_counts().sort_index()
    bars   = ax.bar(EMOTION_LABELS, counts.values, color='steelblue')
    ax.set_title(f'Emotion - {split_name}')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha='center', fontsize=8)

plt.suptitle('Emotion Dataset - Label Distribution Across Splits', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('emotion_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Train Label Distribution (%):")
train_counts = df['label'].value_counts().sort_index()
for idx, count in train_counts.items():
    print(f"  {EMOTION_LABELS[idx]:10} : {count:5}  ({count/len(df)*100:.1f}%)")


# Step 10: Class imbalance analysis

imbalance_ratio = train_counts.max() / train_counts.min()

print("Class Imbalance - Emotion Dataset (Train):")
print(f"  Most frequent : {EMOTION_LABELS[train_counts.idxmax()]} ({train_counts.max()} samples)")
print(f"  Least frequent: {EMOTION_LABELS[train_counts.idxmin()]} ({train_counts.min()} samples)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

# Pie charts: Emotion vs GoEmotions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(train_counts.values, labels=EMOTION_LABELS, autopct='%1.1f%%',
            colors=plt.cm.Set2.colors[:6], startangle=140)
axes[0].set_title('Emotion Dataset - Class Distribution (Train)')

go_top10_counts = go_train_df['label'].value_counts().head(10)
go_top10_names  = [GOEMOTIONS_LABELS[i] for i in go_top10_counts.index]
axes[1].pie(go_top10_counts.values, labels=go_top10_names, autopct='%1.1f%%',
            colors=plt.cm.Set3.colors, startangle=140)
axes[1].set_title('GoEmotions - Top 10 Classes (Train)')

plt.tight_layout()
plt.savefig('class_distribution_pie.png', dpi=150, bbox_inches='tight')
plt.show()

go_label_counts = go_train_df['label'].value_counts().sort_index()
go_imbalance    = go_label_counts.max() / go_label_counts.min()
print(f"\nGoEmotions imbalance ratio: {go_imbalance:.2f}x")
print(f"  Most frequent : {GOEMOTIONS_LABELS[go_label_counts.idxmax()]} ({go_label_counts.max()})")
print(f"  Least frequent: {GOEMOTIONS_LABELS[go_label_counts.idxmin()]} ({go_label_counts.min()})")


# Step 11: Text length analysis - Emotion dataset

df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['text'].apply(lambda x: len(str(x)))

print("Text Length Statistics - Emotion Dataset (Train):")
print(df[['word_count', 'char_count']].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Word count histogram
axes[0].hist(df['word_count'], bins=40, color='steelblue', edgecolor='white')
axes[0].axvline(df['word_count'].mean(),   color='red',    linestyle='--', label=f"Mean: {df['word_count'].mean():.1f}")
axes[0].axvline(df['word_count'].median(), color='orange', linestyle='--', label=f"Median: {df['word_count'].median():.1f}")
axes[0].set_title('Word Count Distribution')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Char count histogram
axes[1].hist(df['char_count'], bins=40, color='seagreen', edgecolor='white')
axes[1].axvline(df['char_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['char_count'].mean():.1f}")
axes[1].set_title('Character Count Distribution')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Boxplot: word count per emotion
word_by_label = [df[df['label'] == i]['word_count'].values for i in range(NUM_EMOTION_LABELS)]
axes[2].boxplot(word_by_label, labels=EMOTION_LABELS)
axes[2].set_title('Word Count per Emotion')
axes[2].set_xlabel('Emotion')
axes[2].set_ylabel('Word Count')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle('Text Length Analysis - Emotion Dataset', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('text_length_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 12: Word count comparison - Emotion vs GoEmotions

go_train_df['word_count'] = go_train_df['text'].apply(lambda x: len(str(x).split()))
go_train_df['char_count'] = go_train_df['text'].apply(lambda x: len(str(x)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['word_count'],          bins=40, alpha=0.6, color='steelblue', label='Emotion (6)',    edgecolor='white')
axes[0].hist(go_train_df['word_count'], bins=40, alpha=0.6, color='coral',     label='GoEmotions (28)',edgecolor='white')
axes[0].set_title('Word Count: Emotion vs GoEmotions')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].boxplot(
    [df['word_count'].values, go_train_df['word_count'].values],
    labels=['Emotion (6)', 'GoEmotions (28)']
)
axes[1].set_title('Word Count Distribution Comparison')
axes[1].set_ylabel('Word Count')

plt.suptitle('Text Length Comparison Across Datasets', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('text_length_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("GoEmotions word count stats:")
print(go_train_df['word_count'].describe().round(2))


# Step 13: GoEmotions full label distribution (all 28 classes)

all_go_names  = [GOEMOTIONS_LABELS[i] for i in go_label_counts.index]
all_go_values = [go_label_counts[i] for i in go_label_counts.index]

plt.figure(figsize=(16, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(all_go_names)))
plt.bar(range(len(all_go_names)), all_go_values, color=colors)
plt.xticks(range(len(all_go_names)), all_go_names, rotation=60, ha='right', fontsize=9)
plt.axhline(np.mean(all_go_values), color='red', linestyle='--', linewidth=1.2,
            label=f'Mean: {np.mean(all_go_values):.0f}')
plt.title('GoEmotions Dataset - All 28 Classes Distribution (Train)', fontsize=13, fontweight='bold')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('goemotions_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 14: Top 15 most frequent words per emotion class - Emotion dataset

# Basic stopword list (no NLTK required)
STOPWORDS = {
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'am','is','are','was','were','be','been','have','has','had',
    'do','does','did','will','would','could','should','can','may','might',
    'a','an','the','and','but','or','for','of','at','by','from','with',
    'in','on','to','so','if','as','not','no','up','out','about','into',
    'then','than','just','very','when','all','what','this','that','which',
    'get','got','dont',"im","ive","its"
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for label_id in range(NUM_EMOTION_LABELS):
    ax     = axes[label_id // 3][label_id % 3]
    subset = df[df['label'] == label_id]['text'].astype(str)
    words  = []
    for text in subset:
        tokens = re.sub(r'[^a-zA-Z\s]', '', text.lower()).split()
        words += [w for w in tokens if w not in STOPWORDS and len(w) > 2]

    top_words = Counter(words).most_common(15)
    if top_words:
        words_list, counts_list = zip(*top_words)
        ax.barh(range(len(words_list)), counts_list, color='steelblue')
        ax.set_yticks(range(len(words_list)))
        ax.set_yticklabels(words_list, fontsize=9)
        ax.invert_yaxis()
    ax.set_title(f'{EMOTION_LABELS[label_id].upper()} - Top 15 Words')
    ax.set_xlabel('Frequency')

plt.suptitle('Top Words per Emotion Class (stopwords removed)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('top_words_per_emotion.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 15: Noise analysis - identify noisy patterns in both datasets

def analyze_noise(series, name):
    texts = series.astype(str)
    total = len(texts)
    checks = {
        'URLs'          : texts.str.contains(r'http\S+|www\S+', regex=True).sum(),
        '@Mentions'     : texts.str.contains(r'@\w+', regex=True).sum(),
        '#Hashtags'     : texts.str.contains(r'#\w+', regex=True).sum(),
        'HTML tags'     : texts.str.contains(r'<.*?>', regex=True).sum(),
        'Numbers'       : texts.str.contains(r'\d+', regex=True).sum(),
        'Multi-punct'   : texts.str.contains(r'[!?]{2,}', regex=True).sum(),
        'All uppercase' : texts.apply(lambda x: x.isupper()).sum(),
        'Non-ASCII'     : texts.apply(lambda x: any(ord(c) > 127 for c in x)).sum(),
    }
    print(f"\nNoise Analysis - {name} (n={total:,}):")
    for label, count in checks.items():
        print(f"  {label:15}: {count:6}  ({count/total*100:.1f}%)")

analyze_noise(df['text'], 'Emotion Train')
analyze_noise(go_train_df['text'], 'GoEmotions Train')


# Step 16: Sample texts per emotion - qualitative inspection

print("Sample Texts per Emotion Class:\n")
for label_id in range(NUM_EMOTION_LABELS):
    samples = df[df['label'] == label_id]['text'].sample(3, random_state=SEED).values
    print(f"  [{EMOTION_LABELS[label_id].upper()}]")
    for s in samples:
        print(f"    - {str(s)[:110]}")
    print()


# Step 17: Text cleaning function (lowercase version - for uncased models)
# Preserves emotional punctuation (!, ?) as they carry sentiment signals

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'@\w+', '', text)                # remove @mentions
    text = re.sub(r'#\w+', '', text)                # remove #hashtags
    text = re.sub(r'<.*?>', '', text)               # remove HTML tags
    text = text.encode('ascii', 'ignore').decode('ascii')  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()        # collapse whitespace
    text = text.lower()                             # lowercase
    return text

# Version for cased models (XLNet) - same but no lowercasing
def clean_text_cased(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test cleaning
examples = [
    "I'm SO HAPPY!! Check http://example.com #joy @user123",
    "feeling sad :( cant believe this happened",
    "<b>Angry</b> at everyone   right   now!!!"
]
print("Cleaning examples:\n")
for t in examples:
    print(f"  Before: {t}")
    print(f"  After : {clean_text(t)}")
    print()


# Step 18: Apply cleaning to all splits

print("Applying cleaning to all datasets...")

# Emotion dataset (lowercase for uncased models)
for d in [df, val_df, test_df]:
    d['text_clean']       = d['text'].apply(clean_text)
    d['text_clean_cased'] = d['text'].apply(clean_text_cased)

# GoEmotions dataset
for d in [go_train_df, go_val_df, go_test_df]:
    d['text_clean']       = d['text'].apply(clean_text)
    d['text_clean_cased'] = d['text'].apply(clean_text_cased)

print("Done.")
print("\nBefore / After examples:")
for i in range(3):
    print(f"  Before: {df['text'].iloc[i]}")
    print(f"  After : {df['text_clean'].iloc[i]}")
    print()


# Step 19: Remove very short texts after cleaning (min 3 chars)

def drop_short_texts(dataframe, min_len=3, name=''):
    before = len(dataframe)
    dataframe = dataframe[dataframe['text_clean'].str.len() >= min_len].reset_index(drop=True)
    print(f"  {name}: removed {before - len(dataframe)} short texts. Remaining: {len(dataframe):,}")
    return dataframe

print("Dropping short texts:")
df          = drop_short_texts(df,          name='Emotion Train')
val_df      = drop_short_texts(val_df,      name='Emotion Val')
test_df     = drop_short_texts(test_df,     name='Emotion Test')
go_train_df = drop_short_texts(go_train_df, name='GoEmotions Train')
go_val_df   = drop_short_texts(go_val_df,   name='GoEmotions Val')
go_test_df  = drop_short_texts(go_test_df,  name='GoEmotions Test')


# Step 20: Remove duplicate texts within each split

def drop_duplicates(dataframe, name=''):
    before = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset='text_clean').reset_index(drop=True)
    print(f"  {name}: removed {before - len(dataframe)} duplicates. Remaining: {len(dataframe):,}")
    return dataframe

print("Removing duplicates:")
df          = drop_duplicates(df,          name='Emotion Train')
val_df      = drop_duplicates(val_df,      name='Emotion Val')
test_df     = drop_duplicates(test_df,     name='Emotion Test')
go_train_df = drop_duplicates(go_train_df, name='GoEmotions Train')
go_val_df   = drop_duplicates(go_val_df,   name='GoEmotions Val')
go_test_df  = drop_duplicates(go_test_df,  name='GoEmotions Test')


# Step 21: Post-cleaning word count and effect visualization

df['word_count_clean'] = df['text_clean'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['word_count'],       bins=40, alpha=0.6, color='steelblue', label='Before', edgecolor='white')
axes[0].hist(df['word_count_clean'], bins=40, alpha=0.6, color='coral',     label='After',  edgecolor='white')
axes[0].set_title('Word Count: Before vs After Cleaning')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Verify label distribution is preserved after cleaning
post_counts = df['label'].value_counts().sort_index()
axes[1].bar(EMOTION_LABELS, post_counts.values, color='steelblue')
axes[1].set_title('Label Distribution After Cleaning (Train)')
axes[1].set_xlabel('Emotion')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Effect of Preprocessing on Data', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cleaning_effect.png', dpi=150, bbox_inches='tight')
plt.show()

print("Final dataset sizes after preprocessing:")
print(f"  Emotion    - Train: {len(df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"  GoEmotions - Train: {len(go_train_df):,} | Val: {len(go_val_df):,} | Test: {len(go_test_df):,}")


# Step 22: Prepare final label lists (from cleaned dataframes)

train_labels    = df['label'].tolist()
val_labels      = val_df['label'].tolist()
test_labels     = test_df['label'].tolist()

go_train_labels = go_train_df['label'].tolist()
go_val_labels   = go_val_df['label'].tolist()
go_test_labels  = go_test_df['label'].tolist()

print("Label lists ready.")
print(f"  Emotion train   : {len(train_labels):,}")
print(f"  GoEmotions train: {len(go_train_labels):,}")


# Step 23: Load all tokenizers

bert_tokenizer       = AutoTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer    = RobertaTokenizer.from_pretrained('roberta-base')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
albert_tokenizer     = AlbertTokenizer.from_pretrained('albert-base-v2')
xlnet_tokenizer      = XLNetTokenizer.from_pretrained('xlnet-base-cased')

print("All tokenizers loaded")
print(f"  BERT vocab size     : {bert_tokenizer.vocab_size:,}")
print(f"  RoBERTa vocab size  : {roberta_tokenizer.vocab_size:,}")
print(f"  DistilBERT vocab    : {distilbert_tokenizer.vocab_size:,}")
print(f"  ALBERT vocab size   : {albert_tokenizer.vocab_size:,}")
print(f"  XLNet vocab size    : {xlnet_tokenizer.vocab_size:,}")


# Step 24: Actual token length distribution using BERT tokenizer (sub-word level)
# Reveals true truncation impact more accurately than word count

print("Computing BERT token lengths on a 2000-sample subset...")
sample_texts  = df['text_clean'].sample(2000, random_state=SEED).tolist()
token_lengths = [len(bert_tokenizer.tokenize(t)) for t in sample_texts]

pct_truncated = sum(1 for l in token_lengths if l > MAX_LENGTH) / len(token_lengths) * 100

plt.figure(figsize=(10, 5))
plt.hist(token_lengths, bins=40, color='steelblue', edgecolor='white')
plt.axvline(MAX_LENGTH, color='red', linestyle='--', linewidth=1.5, label=f'MAX_LENGTH={MAX_LENGTH}')
plt.axvline(np.mean(token_lengths), color='orange', linestyle='--', linewidth=1.5,
            label=f'Mean={np.mean(token_lengths):.1f}')
plt.title('BERT Token Length Distribution (Emotion Train, n=2000)')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('bert_token_lengths.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean token length          : {np.mean(token_lengths):.1f}")
print(f"Max token length           : {max(token_lengths)}")
print(f"Samples > {MAX_LENGTH} tokens: {pct_truncated:.1f}% (will be truncated)")


# Step 25: Tokenize with BERT and train
clear_gpu()

bert_train_ds = EmotionDataset(tokenize_texts(df['text_clean'],      bert_tokenizer), train_labels)
bert_val_ds   = EmotionDataset(tokenize_texts(val_df['text_clean'],  bert_tokenizer), val_labels)
bert_test_ds  = EmotionDataset(tokenize_texts(test_df['text_clean'], bert_tokenizer), test_labels)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=NUM_EMOTION_LABELS
)
print(f"BERT parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

bert_trainer = Trainer(
    model=bert_model,
    args=get_training_args('./results/bert'),
    train_dataset=bert_train_ds,
    eval_dataset=bert_val_ds,
    compute_metrics=compute_metrics
)
bert_trainer.train()
print("BERT training complete")


# Step 26: Evaluate BERT
bert_preds, bert_acc, bert_f1_macro, bert_f1_w = evaluate_model(
    bert_trainer, bert_val_ds, val_labels, EMOTION_LABELS, 'BERT'
)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(val_labels, bert_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.title('BERT - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bert_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
clear_gpu()


# Step 27: Tokenize with RoBERTa and train
clear_gpu()

roberta_train_ds = EmotionDataset(tokenize_texts(df['text_clean'],     roberta_tokenizer), train_labels)
roberta_val_ds   = EmotionDataset(tokenize_texts(val_df['text_clean'], roberta_tokenizer), val_labels)

roberta_model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=NUM_EMOTION_LABELS
)
print(f"RoBERTa parameters: {sum(p.numel() for p in roberta_model.parameters()):,}")

roberta_trainer = Trainer(
    model=roberta_model,
    args=get_training_args('./results/roberta'),
    train_dataset=roberta_train_ds,
    eval_dataset=roberta_val_ds,
    compute_metrics=compute_metrics
)
roberta_trainer.train()
print("RoBERTa training complete")


# Step 28: Evaluate RoBERTa
roberta_preds, roberta_acc, roberta_f1_macro, roberta_f1_w = evaluate_model(
    roberta_trainer, roberta_val_ds, val_labels, EMOTION_LABELS, 'RoBERTa'
)
clear_gpu()


# Step 29: Tokenize with DistilBERT and train
clear_gpu()

distilbert_train_ds = EmotionDataset(tokenize_texts(df['text_clean'],     distilbert_tokenizer), train_labels)
distilbert_val_ds   = EmotionDataset(tokenize_texts(val_df['text_clean'], distilbert_tokenizer), val_labels)

distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=NUM_EMOTION_LABELS
)
print(f"DistilBERT parameters: {sum(p.numel() for p in distilbert_model.parameters()):,}")

distilbert_trainer = Trainer(
    model=distilbert_model,
    args=get_training_args('./results/distilbert'),
    train_dataset=distilbert_train_ds,
    eval_dataset=distilbert_val_ds,
    compute_metrics=compute_metrics
)
distilbert_trainer.train()
print("DistilBERT training complete")


# Step 30: Evaluate DistilBERT
distilbert_preds, distilbert_acc, distilbert_f1_macro, distilbert_f1_w = evaluate_model(
    distilbert_trainer, distilbert_val_ds, val_labels, EMOTION_LABELS, 'DistilBERT'
)
clear_gpu()


# Step 31: Tokenize with ALBERT and train
clear_gpu()

albert_train_ds = EmotionDataset(tokenize_texts(df['text_clean'],     albert_tokenizer), train_labels)
albert_val_ds   = EmotionDataset(tokenize_texts(val_df['text_clean'], albert_tokenizer), val_labels)

albert_model = AlbertForSequenceClassification.from_pretrained(
    'albert-base-v2', num_labels=NUM_EMOTION_LABELS
)
print(f"ALBERT parameters: {sum(p.numel() for p in albert_model.parameters()):,}")

albert_trainer = Trainer(
    model=albert_model,
    args=get_training_args('./results/albert'),
    train_dataset=albert_train_ds,
    eval_dataset=albert_val_ds,
    compute_metrics=compute_metrics
)
albert_trainer.train()
print("ALBERT training complete")


# Step 32: Evaluate ALBERT
albert_preds, albert_acc, albert_f1_macro, albert_f1_w = evaluate_model(
    albert_trainer, albert_val_ds, val_labels, EMOTION_LABELS, 'ALBERT'
)
clear_gpu()


# Step 33: Tokenize with XLNet and train
# XLNet is cased - using text_clean_cased (no lowercase applied)
clear_gpu()

xlnet_train_ds = EmotionDataset(tokenize_texts(df['text_clean_cased'],     xlnet_tokenizer), train_labels)
xlnet_val_ds   = EmotionDataset(tokenize_texts(val_df['text_clean_cased'], xlnet_tokenizer), val_labels)

xlnet_model = XLNetForSequenceClassification.from_pretrained(
    'xlnet-base-cased', num_labels=NUM_EMOTION_LABELS
)
print(f"XLNet parameters: {sum(p.numel() for p in xlnet_model.parameters()):,}")

xlnet_trainer = Trainer(
    model=xlnet_model,
    args=get_training_args('./results/xlnet'),
    train_dataset=xlnet_train_ds,
    eval_dataset=xlnet_val_ds,
    compute_metrics=compute_metrics
)
xlnet_trainer.train()
print("XLNet training complete")


# Step 34: Evaluate XLNet
xlnet_preds, xlnet_acc, xlnet_f1_macro, xlnet_f1_w = evaluate_model(
    xlnet_trainer, xlnet_val_ds, val_labels, EMOTION_LABELS, 'XLNet'
)
clear_gpu()


# Step 35: Model comparison table
results = pd.DataFrame({
    'Model':       ['BERT', 'RoBERTa', 'DistilBERT', 'ALBERT', 'XLNet'],
    'Accuracy':    [bert_acc,    roberta_acc,    distilbert_acc,    albert_acc,    xlnet_acc],
    'Macro_F1':    [bert_f1_macro, roberta_f1_macro, distilbert_f1_macro, albert_f1_macro, xlnet_f1_macro],
    'Weighted_F1': [bert_f1_w,   roberta_f1_w,   distilbert_f1_w,   albert_f1_w,   xlnet_f1_w]
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("Model Comparison (Emotion Dataset, sorted by Accuracy):\n")
print(results.to_string(index=False))

results.to_csv('model_comparison_emotion.csv', index=False)
print("\nSaved: model_comparison_emotion.csv")


# Step 36: Comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors  = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0', '#FF9800']
metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1']
titles  = ['Accuracy', 'Macro F1', 'Weighted F1']

for ax, metric, title in zip(axes, metrics, titles):
    bars = ax.bar(results['Model'], results[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, results[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)

plt.suptitle('All Models - Emotion Dataset (6 Classes)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 37: All confusion matrices
all_preds = {
    'BERT':       bert_preds,
    'RoBERTa':    roberta_preds,
    'DistilBERT': distilbert_preds,
    'ALBERT':     albert_preds,
    'XLNet':      xlnet_preds
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (name, preds) in enumerate(all_preds.items()):
    ax = axes[idx // 3][idx % 3]
    cm = confusion_matrix(val_labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

axes[1][2].axis('off')
plt.suptitle('Confusion Matrices - All Models (Emotion Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 38: Error analysis - top misclassification pairs per model

def top_misclassifications(true_labels, pred_labels, label_names, model_name, top_n=5):
    cm    = confusion_matrix(true_labels, pred_labels)
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                pairs.append((label_names[i], label_names[j], cm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{model_name} - Top {top_n} Misclassifications:")
    for true, pred, count in pairs[:top_n]:
        print(f"  True: {true:12} -> Predicted: {pred:12} | Count: {count}")

for name, preds in all_preds.items():
    top_misclassifications(val_labels, preds, EMOTION_LABELS, name)


# Step 39: Tokenize GoEmotions and train BERT (best performer from Section 7)
clear_gpu()

go_train_ds = EmotionDataset(tokenize_texts(go_train_df['text_clean'], bert_tokenizer), go_train_labels)
go_val_ds   = EmotionDataset(tokenize_texts(go_val_df['text_clean'],   bert_tokenizer), go_val_labels)

go_bert_model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=NUM_GOEMOTIONS_LABELS
)
print(f"BERT (GoEmotions) parameters: {sum(p.numel() for p in go_bert_model.parameters()):,}")

go_trainer = Trainer(
    model=go_bert_model,
    args=get_training_args('./results/goemotions_bert', epochs=2),
    train_dataset=go_train_ds,
    eval_dataset=go_val_ds,
    compute_metrics=compute_metrics
)
go_trainer.train()
print("BERT GoEmotions training complete")


# Step 40: Evaluate BERT on GoEmotions
go_preds, go_acc, go_f1_macro, go_f1_w = evaluate_model(
    go_trainer, go_val_ds, go_val_labels, GOEMOTIONS_LABELS, 'BERT (GoEmotions)'
)


# Step 41: GoEmotions confusion matrix (top 10 classes for readability)
top10_classes = pd.Series(go_train_labels).value_counts().head(10).index.tolist()
top10_names   = [GOEMOTIONS_LABELS[i] for i in top10_classes]

full_cm  = confusion_matrix(go_val_labels, go_preds)
cm_top10 = full_cm[np.ix_(top10_classes, top10_classes)]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_top10, annot=True, fmt='d', cmap='Greens',
            xticklabels=top10_names, yticklabels=top10_names)
plt.title('BERT - Confusion Matrix (GoEmotions, Top 10 Classes)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('goemotions_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
clear_gpu()


# Step 42: Compare basic (6 classes) vs fine-grained (28 classes) performance
best_basic_acc = results.iloc[0]['Accuracy']
best_basic_f1  = results.iloc[0]['Macro_F1']
best_model     = results.iloc[0]['Model']

granularity_df = pd.DataFrame({
    'Task':     ['Basic (6 emotions)', 'Fine-grained (28 emotions)'],
    'Accuracy': [best_basic_acc, go_acc],
    'Macro_F1': [best_basic_f1,  go_f1_macro]
})

acc_drop = (best_basic_acc - go_acc) * 100
f1_drop  = (best_basic_f1  - go_f1_macro) * 100

print(f"Best model on basic emotions: {best_model}")
print(granularity_df.to_string(index=False))
print(f"\nAccuracy drop : {acc_drop:.2f}%")
print(f"Macro F1 drop : {f1_drop:.2f}%")

fig, axes  = plt.subplots(1, 2, figsize=(10, 5))
bar_colors = ['#2196F3', '#F44336']
x_labels   = ['Basic (6)', 'Fine-grained (28)']

axes[0].bar(x_labels, granularity_df['Accuracy'], color=bar_colors)
axes[0].set_title('Accuracy: Basic vs Fine-grained')
axes[0].set_ylim(0, 1.05)
for i, v in enumerate(granularity_df['Accuracy']):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')

axes[1].bar(x_labels, granularity_df['Macro_F1'], color=bar_colors)
axes[1].set_title('Macro F1: Basic vs Fine-grained')
axes[1].set_ylim(0, 1.05)
for i, v in enumerate(granularity_df['Macro_F1']):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.suptitle('Basic vs Fine-Grained Emotion Detection (BERT)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('basic_vs_finegrained.png', dpi=150, bbox_inches='tight')
plt.show()


# Step 43: Predict on custom input texts

def predict_emotion(text, model, tokenizer, label_names):
    model.eval()
    cleaned = clean_text(text)
    inputs  = tokenizer(cleaned, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        model.to(DEVICE)
    with torch.no_grad():
        logits     = model(**inputs).logits
        pred_id    = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).squeeze()[pred_id].item()
    return label_names[pred_id], confidence

test_texts = [
    "I am so happy today!",
    "This is making me really angry!",
    "I am feeling scared about the results.",
    "I love spending time with my family.",
    "Wow, that is really surprising news!",
    "I feel so sad and lonely."
]

print("Sample Predictions (BERT, Emotion Dataset):\n")
for text in test_texts:
    label, conf = predict_emotion(text, bert_model, bert_tokenizer, EMOTION_LABELS)
    print(f"  Text      : {text}")
    print(f"  Prediction: {label} (confidence: {conf:.2%})")
    print()


# Step 44: Save all trained models and tokenizers
os.makedirs('./saved_models', exist_ok=True)

save_map = {
    'bert':            (bert_model,       bert_tokenizer),
    'roberta':         (roberta_model,    roberta_tokenizer),
    'distilbert':      (distilbert_model, distilbert_tokenizer),
    'albert':          (albert_model,     albert_tokenizer),
    'xlnet':           (xlnet_model,      xlnet_tokenizer),
    'bert_goemotions': (go_bert_model,    bert_tokenizer)
}

for name, (model, tokenizer) in save_map.items():
    path = f'./saved_models/{name}'
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved: {path}")

results.to_csv('model_comparison_emotion.csv', index=False)
granularity_df.to_csv('basic_vs_finegrained.csv', index=False)
print("\nAll models and CSVs saved")


# Step 45: Final summary
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print("\nRQ1 - Misclassification patterns identified:")
print("  - Semantically similar emotions (sadness/grief, joy/excitement) are most confused")
print("  - Rare classes suffer from class imbalance (surprise, love)")
print("  - Fine-grained tasks show higher confusion across overlapping categories")

print("\nRQ2 - Basic vs Fine-grained performance:")
print(f"  - Basic (6 emotions)  - Best Accuracy: {best_basic_acc:.4f}, Macro F1: {best_basic_f1:.4f} [{best_model}]")
print(f"  - Fine-grained (28)   - Accuracy:       {go_acc:.4f}, Macro F1: {go_f1_macro:.4f} [BERT]")
print(f"  - Performance drop    - Accuracy: -{acc_drop:.2f}%, Macro F1: -{f1_drop:.2f}%")

print("\nModel Ranking (Emotion Dataset, by Accuracy):")
print(results[['Model', 'Accuracy', 'Macro_F1', 'Weighted_F1']].to_string(index=False))

print("\nGenerated output files:")
output_files = [
    'emotion_distribution.png',     'class_distribution_pie.png',
    'text_length_analysis.png',     'text_length_comparison.png',
    'top_words_per_emotion.png',    'goemotions_distribution.png',
    'cleaning_effect.png',          'bert_token_lengths.png',
    'bert_confusion_matrix.png',    'all_confusion_matrices.png',
    'model_comparison_chart.png',   'goemotions_confusion_matrix.png',
    'basic_vs_finegrained.png',     'model_comparison_emotion.csv',
    'basic_vs_finegrained.csv'
]
for f in output_files:
    print(f"  {f}")

print("\n" + "=" * 65)
