import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import random
import torch

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
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        randomnum = opt.rand  # 0.5
        model.set_input(data, randomnum, 0)  # def set_input(input, beta, alpha)
        model.test()
        visuals = model.get_current_visuals()  # [('fake_B', self.fake_B)]
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path), 'z: ', randomnum)

        visualizer.save_images_withlabel(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()
