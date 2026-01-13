import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure directory exists
if not os.path.exists('models'):
    os.makedirs('models')

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval() # Set to evaluation mode

# Dummy input for the exporter
dummy_input = tokenizer("MLOps is fun!", return_tensors="pt")

torch.onnx.export(
    model, 
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "models/roberta.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence_length'}},
    opset_version=14  # <--- CHANGED FROM 12 TO 14
)
print("✅ Created models/roberta.onnx with Opset 14")