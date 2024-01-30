import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import torch
from torch.nn import DataParallel
import random

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Fine-tuning을 위한 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        input_text = f"question: {entry['question']} context: {' '.join(entry['documents'])}"
        target_text = entry["answer"]

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        targets = self.tokenizer(target_text, return_tensors="pt", max_length=1024, truncation=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Padding
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def split_train_validation_data(data, validation_ratio=0.1, random_seed=42):
    random.seed(random_seed)
    data_size = len(data)
    validation_size = int(data_size * validation_ratio)
    validation_indices = random.sample(range(data_size), validation_size)
    
    train_data = [data[i] for i in range(data_size) if i not in validation_indices]
    validation_data = [data[i] for i in validation_indices]
    
    return train_data, validation_data

# 데이터 로딩 및 전처리
with open("release_train_data.json", "r") as json_file:
    all_train_data = json.load(json_file)

# 학습 데이터와 검증 데이터 분리
train_data, validation_data = split_train_validation_data(all_train_data, validation_ratio=0.1)

# 토크나이저 및 모델 로드
model_name = "t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(model_name, device_map = "balanced",  max_memory={0: "20GB", 1: "20GB"})
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map = "balanced",  max_memory={0: "20GB", 1: "20GB"})

# GPU가 여러 개인 경우

# Fine-tuning을 위한 데이터셋 및 데이터로더 생성
train_dataset = CustomDataset(train_data, tokenizer)
validation_dataset = CustomDataset(validation_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# 모델 및 훈련 설정
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

# Early Stopping 및 모델 저장 설정
best_validation_loss = float('inf')
patience = 5  # 몇 번의 epoch동안 개선이 없을 때 학습을 중지할지 결정
counter = 0

num_epochs = 50
for epoch in range(num_epochs):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = torch.sum(outputs.loss)
        
        loss.backward()
        optimizer.step()

    # Validation
    validation_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc=f"Validation - Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) 

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            validation_loss += outputs.loss.sum().item()

    validation_loss /= len(validation_loader)

    print(f"Validation Loss: {validation_loss}")

    # Early Stopping 및 모델 저장
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        counter = 0
        # Save the model
        model.save_pretrained("fine_tuned_t5_base")
        tokenizer.save_pretrained("fine_tuned_t5_base")
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping after {epoch+1} epochs without improvement.")
        break

print("Training finished.")
print("=" * 100)
print("\n\n\n")


# 예측 생성
def generate_predictions_t5(data, model, tokenizer):
    model.to(device)
    model.eval()
    predictions = []

    for entry in tqdm(data, desc="Generating predictions"):
        uuid = entry["uuid"]
        question = entry["question"]
        
        # history 및 document 전처리 (emoji 제거)
        
        document = ' '.join(remove_emoji(doc) for doc in entry["documents"])
        
        # T5 모델에 입력을 맞춤
        input_text = f"question: {question} context: {document}"
        
        input_ids = tokenizer(input_text, truncation=True, max_length=2048).input_ids
        
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # GPU로 이동

        # 예측 생성
        output = model.generate(input_ids, max_length=200, num_beams=10, length_penalty=2.0, early_stopping=True)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        # 결과 저장
        predictions.append({"uuid": uuid, "prediction": prediction})

    return predictions

# 학습된 모델 로드
fine_tuned_model = T5ForConditionalGeneration.from_pretrained("t5_nohistory_1024_large_early_stop")
fine_tuned_tokenizer = T5Tokenizer.from_pretrained("t5_nohistory_1024_large_early_stop")

# JSON 파일에서 데이터 불러오기
with open("release_phase1_eval_data_wo_gt.json", "r") as json_file:
    data = json.load(json_file)

# 예측 생성
predictions_t5 = generate_predictions_t5(data, fine_tuned_model, fine_tuned_tokenizer)

# 결과를 JSON 파일로 저장
with open("submission.json", "w") as output_file:
    json.dump(predictions_t5, output_file, ensure_ascii=False, indent=2)

print("Predictions saved to submission.json")
