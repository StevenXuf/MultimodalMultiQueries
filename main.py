import timm
import torch
import torchvision.transforms as transforms

from datasets import load_dataset,Audio
from PIL import Image
from transformers import AutoModel, AutoTokenizer,HubertModel,AutoProcessor

def extract_vision_features(model,image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    image_transform=get_transforms(model)

    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model(image_tensor).squeeze().cpu().numpy()
    
    return image_features

def extract_language_features(model,tokenizer,text):
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_features = model(**encoded).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return text_features

def extract_features(example,vision_model,language_model):
    image = example['image']
    image_features = extract_vision_features(vision_model,image)

    text = str(example['text'])  # Make sure label is string
    text_features = extract_language_features(language_model,text)

    return {
        "image_features": image_features,
        "text_features": text_features
    }

def get_transforms(model):
    config = model.default_cfg
    transform = transforms.Compose([
        transforms.Resize(config['input_size'][-1]),
        transforms.CenterCrop(config['input_size'][-1]),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])
    return transform

def predict_image(model, image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(model)
    input_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    return top5_prob, top5_catid

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def map_to_array(batch):
    batch["array"] = batch['audio']['array']
    return batch

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentence_bert_name='sentence-transformers/all-mpnet-base-v2'
    hubert_model_name='facebook/hubert-large-ls960-ft'
    dataset_name='anjunhu/naively_captioned_CUB2002011_test'
    sampling_rate=16000

    #cub = load_dataset(dataset_name)
    
    sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]

    vit_extractor= timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).to(device)
    
    print(f"\nFeature extraction:")
    print(f"ViT features shape: {vit_extractor(torch.randn(1, 3, 224, 224)).shape}")


    sentence_bert=AutoModel.from_pretrained(sentence_bert_name)
    tokenizer_sentence_bert=AutoTokenizer.from_pretrained(sentence_bert_name)
    # Tokenize sentences
    encoded_input = tokenizer_sentence_bert(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        output_sentence_bert= sentence_bert(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(output_sentence_bert, encoded_input['attention_mask'])

    print("Sentence embeddings:")
    print(sentence_embeddings.size())
    
    processor_hubert=AutoProcessor.from_pretrained(hubert_model_name)
    hubert = HubertModel.from_pretrained(hubert_model_name,torch_dtype=torch.float16).to(device)
    
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    ds = ds.map(map_to_array)

    input_values = processor_hubert(ds["array"][0], return_tensors="pt"i,sampling_rate=sampling_rate).input_values.to(torch.float16).to(device)
    hidden_states = hubert(input_values).last_hidden_state.mean(dim=1)
    print(hidden_states.size())
