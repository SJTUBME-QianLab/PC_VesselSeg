import os
import sys
sys.path.append('./')

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import math


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    # opt.epoch_count 1
    total_steps = (opt.epoch_count-1)*dataset_size  # epoch_count = 1, 所以total_steps是用来记录自己训练多少步？

#    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # opt.progressive_epoch 111
    for epoch in range(opt.epoch_count, opt.progressive_epoch):  # (1, 111)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):

            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            if total_steps % opt.print_freq == 0:  # opt.print_freq=100
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            if total_steps % 5 == 0:
                sign = '1'
            elif total_steps % 5 == 1:
                sign = '1'
            elif total_steps % 5 == 2:
                sign = '1'
            elif total_steps % 5 == 3:
                sign = '1'
            elif total_steps % 5 == 4:
                sign = '2' 
            beta = 1
            # alpha = math.exp((epoch-1.0/2.0*opt.progressive_epoch)/(1.0/4.0*opt.progressive_epoch))
            alpha = math.exp((total_steps-1.0/2.0*(opt.progressive_epoch-1)*dataset_size)/(1.0/4.0*(opt.progressive_epoch-1)*dataset_size))
            # opt.batchSize 1
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data, beta, alpha, sign)

            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:  # opt.display_freq = 400
                save_result = total_steps % opt.update_html_freq == 0  # update_html_freq=1000
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:  # print_freq = 100
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:  # save_latest_freq = 5000
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


