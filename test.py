import os
from PIL import Image
from tqdm import tqdm
from skimage import io

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import *
from model import CKLNet_Encoder, CKLNet_Decoder


def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)

    out = (x - MIN) / (MAX - MIN)

    return out


def save_output(image_name, pred, save_dir):
    predict = pred
    predict = predict.squeeze()
    predict = predict.cpu().data.numpy()
    predict = Image.fromarray(predict * 255).convert('RGB')

    image = io.imread(image_name)
    predict = predict.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    predict.save(save_dir + image_name.split('/')[-1][:-4] + '.png')


if __name__ == '__main__':
    # --------- Define the address and image format ---------
    image_dir = "../Dataset/ESDIs/test/Img/"
    prediction_dir = "./results/ESDIs/"
    model_dir = "./model_save/ESDIs/"
    num_classes = 14 # ESDIs: 14, SD-Saliency-900: 3

    '''
    === ESDIs ===
    0->roll-printing
    1->water_spot
    2->oil_spot
    3->foreign matter inclusion
    4->patches
    5->abrasion mask
    6->iron sheet ash
    7->oxide scale of temperature system
    8->oxide scale of plate system
    9->red iron
    10->slag inclusion
    11->scratches
    12->punching_hole
    13->welding_line

    === SD-Saliency-900 ===
    0->In
    1->Pa
    2->Sc
    '''

    img_name_list = []
    for img_file in os.listdir(image_dir):
        img_file_path = os.path.join(image_dir, img_file)
        img_name_list.append(img_file_path)

    transform_img=transforms.Compose([
        transforms.Resize(336), # Resize
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    # --------- Load the data ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[], transform_img=transform_img)
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=8)

    # --------- Define the model ---------
    encoder = CKLNet_Encoder(num_classes)
    encoder.load_state_dict(torch.load(model_dir+'Encoder.pth'))
    if torch.cuda.is_available():
        encoder.cuda()
    encoder.eval()

    decoder_dict = {}
    for c in range(num_classes):
        decoder = CKLNet_Decoder()
        decoder.load_state_dict(torch.load(f"{model_dir}{c}.pth"))
        decoder_dict[c] = decoder

    # --------- Generate prediction images ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):
        inputs_test, name_list = data_test['image'], data_test['name']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        score1, score2, score3, score4, score5, Cls = encoder(inputs_test)

        # class adapter
        decoder = decoder_dict[int(Cls)]
        if torch.cuda.is_available():
            decoder.cuda()
        decoder.eval()

        pred = decoder(score1, score2, score3, score4, score5)

        # normalization
        pred = pred[0, 0, :, :]
        pred = normPRED(pred)
        save_output(name_list[0], pred, prediction_dir)

        del score1, score2, score3, score4, score5, Cls, pred, decoder
        torch.cuda.ipc_collect()