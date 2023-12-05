import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load and preprocess your data from Excel
data = pd.read_excel("self_label.xlsx")
texts = data['text'].tolist()
labels = data['sentiment'].tolist()

# Map sentiment labels to -1, 0, 1
label_mapping = {-1: 0, 0: 1, 1: 2}
labels = [label_mapping[label] for label in labels]

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load the pre-trained model and tokenizer
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels)
)

# Custom data collator
def custom_data_collator(features):
    inputs = {
        'input_ids': torch.stack([feature[0] for feature in features]),
        'attention_mask': torch.stack([feature[1] for feature in features]),
        'labels': torch.tensor([feature[2] for feature in features])
    }
    return inputs

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./sentiment_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_data_collator,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./saved_model")
