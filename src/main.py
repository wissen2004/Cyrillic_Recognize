from const import FROM_CHECKPOINT_PATH
from config import N_HEADS, ENC_LAYERS, DEC_LAYERS, HIDDEN, DROPOUT
import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE, ALPHABET
import model2

model = model2.TransformerModel2(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                                nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(FROM_CHECKPOINT_PATH))
model.eval()

def process_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image

def predict_image(image_path):
    image = process_image(image_path)
    with torch.no_grad():
        output = model.predict(image)
    decoded_output = ''.join([ALPHABET[idx] for idx in output[0] if idx < len(ALPHABET) and ALPHABET[idx] not in {'SOS', 'EOS'}])
    return decoded_output

if __name__ == "__main__":
    image_path = "C:\\Users\\ASUS\\Desktop\\3kurs\\kursWork\\data\\test\\test96.png"
    result = predict_image(image_path)
    print("Распознанный текст:", result)
