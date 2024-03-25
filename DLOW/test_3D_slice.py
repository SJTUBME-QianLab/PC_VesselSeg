import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import random
import torch
import nrrd

def seed_torch(seed=0):
    print("seed:", seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed_torch(1)
seed_torch(2)
# seed_torch(3)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website  web_dir = ./results/MultiModaltoCT_0.5/test_latest
    web_dir = os.path.join(opt.results_dir, opt.name+'_%s' % opt.rand, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    # filename = os.path.join(opt.results_dir, opt.name, 'random_label.txt')

    save_path = os.path.join(web_dir, "pred")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, data in enumerate(dataset):
        # if i < 10:
        #     continue
        if i >= opt.how_many:  # 162
            print('stop')
            break
        randomnum = opt.rand
        c, h, w = data["A"].shape[-3], data["A"].shape[-2], data["A"].shape[-1]
        pred = np.zeros([c, h, w])
        for j in range(c):
            arr = data["A"][:, j, :, :]
            arr = arr.unsqueeze(0).float()
            temp_data = {'A': arr, 'A_paths': data["A_paths"]}
            with torch.no_grad():
                model.set_input(temp_data, randomnum, 0)  # def set_input(input, beta, alpha)
                model.test()
                visuals = model.get_current_visuals()  # [('fake_B', self.fake_B)]
                temp_pred = visuals["fake_B"]

                temp_img = np.tile(temp_pred[0].cpu().detach().numpy(), (3, 1, 1))
                temp_img = (np.transpose(temp_img, (1, 2, 0)) + 1) / 2.0 * 255.0
                from PIL import Image
                temp_img = Image.fromarray(temp_img.astype(np.uint8))

                temp_path = save_path+'/%s' % i
                # if not os.path.exists(temp_path):
                #     os.makedirs(temp_path)
                temp_img.save(temp_path+'.png')
                temp_img = np.array(temp_img).astype(np.float)
                temp_img = np.transpose(temp_img, (2, 0, 1))
                temp_img = temp_img[0]
                temp_img /= 255
                pred[j, :, :] = temp_img
                print('%04d: process image slice %s... ' % (i, j), 'z: ', randomnum)

        if not os.path.exists(os.path.join(
                web_dir,
                "pred_3D")):
            os.makedirs(os.path.join(
                web_dir,
                "pred_3D"))
        nrrd.write(
            os.path.join(
                web_dir,
                "pred_3D",
                "%s.nrrd" % (i),
            ),
            pred.astype(float).transpose((2, 1, 0)),
        )
        # "%s.nrrd" % (data["A_paths"][0]),
        # fp = open(filename,'a+')
        # fp.write(str(img_path)+"        "+"random_label:"+str(randomnum)+'\n')
        # fp.close()
        with open(
                os.path.join(
                    web_dir,
                    "transfer.txt"
                ),
                "a",
        ) as f:
            f.write('index:%s, z: %s \n' % (str(i), str(randomnum)))
    webpage.save()
