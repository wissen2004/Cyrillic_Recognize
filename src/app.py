import os
import torch
from flask import Flask, request, render_template, redirect
from PIL import Image
from torchvision import transforms
import model2
from const import FROM_CHECKPOINT_PATH
from config import N_HEADS, ENC_LAYERS, DEC_LAYERS, DEVICE, HIDDEN, DROPOUT, ALPHABET

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            result = predict_image(file_path)
    return render_template('upload.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
