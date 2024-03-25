import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np


class CycleGANModel(BaseModel):
    def name(self):
        return "CycleGANModel"

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        opt.use_sigmoid = opt.no_lsgan
        self.nlatent = opt.nlatent
        self.label_intensity = opt.label_intensity
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
            "D_A_classification",
            "D_B_classification",
            "D_A_classification_0",
            "D_B_classification_0",
            "G_A_classification",
            "G_B_classification",
            "G_A_classification_0",
            "G_B_classification_0",
        ]
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_A")
            visual_names_B.append("idt_B")

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = [
                "G_A",
                "G_B",
                "D_A",
                "D_B",
                "D_A_classification",
                "D_B_classification",
                "D_A_classification_0",
                "D_B_classification_0",
                "DeConv",
            ]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B", "DeConv"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
        #                                opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_A = networks.define_stochastic_G(
            nlatent=opt.nlatent,
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=64,
            which_model_netG=opt.which_model_netG,
            norm=opt.norm,
            use_dropout=opt.use_dropout,
            gpu_ids=opt.gpu_ids,
        )
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_stochastic_G(
            nlatent=opt.nlatent,
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=64,
            which_model_netG=opt.which_model_netG,
            norm=opt.norm,
            use_dropout=opt.use_dropout,
            gpu_ids=opt.gpu_ids,
        )
        self.netDeConv = networks.define_InitialDeconv(gpu_ids=self.gpu_ids)
        enc_input_nc = opt.output_nc
        if opt.enc_A_B:
            enc_input_nc += opt.input_nc

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
            #                                opt.which_model_netD,
            #                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_A = networks.define_D_B(
                input_nc=opt.output_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )
            self.netD_B = networks.define_D_B(
                input_nc=opt.output_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )
            self.netD_A_classification = networks.define_D_B(
                input_nc=opt.output_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )
            self.netD_A_classification_0 = networks.define_D_B(
                input_nc=opt.input_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )
            self.netD_B_classification = networks.define_D_B(
                input_nc=opt.input_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )
            self.netD_B_classification_0 = networks.define_D_B(
                input_nc=opt.input_nc,
                ndf=opt.ndf,
                which_model_netD=opt.which_model_netD,
                norm=opt.norm,
                use_sigmoid=opt.use_sigmoid,
                gpu_ids=opt.gpu_ids,
            )

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor
            )
            self.criterionRegression = networks.GANLoss(
                use_lsgan=True, tensor=self.Tensor
            )
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_DeConv = torch.optim.Adam(
                self.netDeConv.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D_classification = torch.optim.Adam(
                itertools.chain(
                    self.netD_A_classification.parameters(),
                    self.netD_B_classification.parameters(),
                ),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D_classification_0 = torch.optim.Adam(
                itertools.chain(
                    self.netD_A_classification_0.parameters(),
                    self.netD_B_classification_0.parameters(),
                ),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_DeConv)
            self.optimizers.append(self.optimizer_D_classification)
            self.optimizers.append(self.optimizer_D_classification_0)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input, beta, alpha, sign):
        AtoB = self.opt.which_direction == "AtoB"
        # input [1, 256, 256]
        input_A = input["A" if AtoB else "B"]
        input_B = input["B" if AtoB else "A"]

        # self.sign = sign
        if self.isTrain:
            # if sign == '0':
            #     self.random_number = 0
            # elif sign == '0.5':
            #     self.random_number = np.random.rand()
            # elif sign == '1':
            #     self.random_number = 1
            # else:
            #     print("Setinput: Error occur when getting the 0, 1, 0to1")
            # self.add_item = np.ones((1, self.nlatent, 1, 1))*self.random_number
            # self.add_item = torch.FloatTensor(self.add_item)
            if sign == "0":
                self.random_number = 0
            elif sign == "1":
                self.random_number = np.random.beta(alpha, beta)
            elif sign == "2":
                self.random_number = 1
            else:
                print("Setinput: Error occur when getting the 0, 1, 0to1")
            self.add_item = self.netDeConv(
                Variable(
                    torch.FloatTensor([self.random_number]).view(1, 1, 1, 1)
                ).cuda()
            )
        else:
            # self.add_item = np.ones((1, self.nlatent, 1, 1))*self.label_intensity
            # self.add_item = torch.FloatTensor(self.add_item)
            self.add_item = self.netDeConv(
                Variable(
                    torch.FloatTensor([self.label_intensity]).view(1, 1, 1, 1)
                ).cuda()
            )
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda()
            input_B = input_B.cuda()
            # self.add_item = self.add_item.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        # self.image_paths = input["A_paths" if AtoB else "B_paths"]

    # def set_input(self, input, beta, alpha, sign):
    #     AtoB = self.opt.which_direction == 'AtoB'
    #     input_A = input['A' if AtoB else 'B']
    #     input_B = input['B' if AtoB else 'A']
    #     # self.sign = sign
    #     if self.isTrain:
    #         # if sign == '0':
    #         #     self.random_number = 0
    #         # elif sign == '0.5':
    #         #     self.random_number = np.random.rand()
    #         # elif sign == '1':
    #         #     self.random_number = 1
    #         # else:
    #         #     print("Setinput: Error occur when getting the 0, 1, 0to1")
    #         # self.add_item = np.ones((1, self.nlatent, 1, 1))*self.random_number
    #         # self.add_item = torch.FloatTensor(self.add_item)
    #         if sign == '0':
    #             self.random_number = 0
    #         elif sign == '1':
    #             self.random_number = np.random.beta(alpha, beta)
    #         elif sign == '2':
    #             self.random_number = 1
    #         else:
    #             print("Setinput: Error occur when getting the 0, 1, 0to1")
    #         self.add_item = self.netDeConv(
    #             Variable(
    #                 torch.FloatTensor([self.random_number]).view(1, 1, 1, 1)
    #             ).cuda()
    #         )
    #     else:
    #         # self.add_item = np.ones((1, self.nlatent, 1, 1))*self.label_intensity
    #         # self.add_item = torch.FloatTensor(self.add_item)
    #         self.add_item = self.netDeConv(
    #             Variable(
    #                 torch.FloatTensor([self.label_intensity]).view(1, 1, 1, 1)
    #             ).cuda()
    #         )
    #     if len(self.gpu_ids) > 0:
    #         input_A = input_A.cuda()
    #         input_B = input_B.cuda()
    #     # if len(self.gpu_ids) > 0:
    #     #     input_A = input_A.cuda(self.gpu_ids[0], async=True)
    #     #     input_B = input_B.cuda(self.gpu_ids[0], async=True)
    #         # self.add_item = self.add_item.cuda(self.gpu_ids[0], async=True)
    #
    #     self.input_A = input_A
    #     self.input_B = input_B
    #     self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        # self.add_item = Variable(self.add_item)

    def test(self):
        # self.add_item = Variable(self.add_item, volatile=True)
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A(self.real_A, self.add_item)
        self.rec_A = self.netG_B(self.fake_B, self.add_item)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B(self.real_B, self.add_item)
        self.rec_B = self.netG_A(self.fake_A, self.add_item)

    def backward_D_basic(self, netD, real, fake, random_number):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * random_number
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_Regbasic(self, netD, real, fake, random_number):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionRegression(pred_real, 1)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionRegression(pred_fake, 0)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * random_number
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.real_A, 1)
        self.loss_D_A = self.backward_D_Regbasic(
            self.netD_A, self.real_B, self.real_A, 1
        )

    def backward_D_B(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.real_B, 1)
        self.loss_D_B = self.backward_D_Regbasic(
            self.netD_B, self.real_A, self.real_B, 1
        )

    def backward_D_A_classification(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_classification = self.backward_D_basic(
            self.netD_A_classification, self.real_B, fake_B, random_number
        )
        # print("backward_D_A_classification:", random_number)

    def backward_D_B_classification(self, random_number):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B_classification = self.backward_D_basic(
            self.netD_B_classification, self.real_A, fake_A, random_number
        )
        # print("backward_D_B_classification:", random_number)

    def backward_D_A_classification_0(self, random_number):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_classification_0 = self.backward_D_basic(
            self.netD_A_classification_0, self.real_A, fake_B, random_number
        )
        # print("backward_D_A_classification_0:", random_number)

    def backward_D_B_classification_0(self, random_number):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B_classification_0 = self.backward_D_basic(
            self.netD_B_classification_0, self.real_B, fake_A, random_number
        )
        # print("backward_D_B_classification_0:", random_number)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity  # 0.5
        lambda_A = self.opt.lambda_A  # 10
        lambda_B = self.opt.lambda_B  # 10
        lambda_GA = self.opt.lambda_GA
        lambda_GB = self.opt.lambda_GB
        lambda_GA_classification = self.opt.lambda_GA_classification
        lambda_GB_classification = self.opt.lambda_GB_classification
        lambda_GA_classification_0 = self.opt.lambda_GA_classification_0
        lambda_GB_classification_0 = self.opt.lambda_GB_classification_0
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            # self.add_item [1,16,1,1]
            self.idt_A = self.netG_A.forward(self.real_B, self.add_item)
            # self.criterionIdt L1loss
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A, self.add_item)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A, self.add_item)
        # self.criterionGAN    nn.MSELoss()
        self.loss_G_A_classification_0 = (
            self.criterionGAN(self.netD_A_classification_0.forward(self.fake_B), True)
            * lambda_GA_classification_0
            * (1 - self.random_number)
        )
        self.loss_G_A = (
            self.criterionRegression(
                self.netD_A.forward(self.fake_B), self.random_number
            )
            * lambda_GA
        )
        self.loss_G_A_classification = (
            self.criterionGAN(self.netD_A_classification.forward(self.fake_B), True)
            * lambda_GA_classification
            * self.random_number
        )

        # GAN loss D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B, self.add_item)
        self.loss_G_B_classification_0 = (
            self.criterionGAN(self.netD_B_classification_0.forward(self.fake_A), True)
            * lambda_GB_classification_0
            * (1 - self.random_number)
        )
        self.loss_G_B = (
            self.criterionRegression(
                self.netD_B.forward(self.fake_A), self.random_number
            )
            * lambda_GB
        )
        self.loss_G_B_classification = (
            self.criterionGAN(self.netD_B_classification.forward(self.fake_A), True)
            * lambda_GB_classification
            * self.random_number
        )

        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B, self.add_item)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A, self.add_item)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
            + self.loss_G_A_classification
            + self.loss_G_B_classification
            + self.loss_G_A_classification_0
            + self.loss_G_B_classification_0
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_DeConv.zero_grad()
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        # gnorm_D = torch.nn.utils.clip_grad_norm(
        #     itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
        #     self.opt.max_gnorm,
        # )
        self.optimizer_D.step()

        ##########################################################

        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        # gnorm_G = torch.nn.utils.clip_grad_norm(
        #     itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
        #     self.opt.max_gnorm,
        # )
        self.optimizer_G.step()

        ##########################################################

        # D_A and D_B
        self.optimizer_D_classification.zero_grad()
        self.backward_D_A_classification(self.random_number)
        self.backward_D_B_classification(self.random_number)
        # gnorm_D_classification = torch.nn.utils.clip_grad_norm(
        #     itertools.chain(
        #         self.netD_A_classification.parameters(),
        #         self.netD_B_classification.parameters(),
        #     ),
        #     self.opt.max_gnorm,
        # )
        self.optimizer_D_classification.step()

        self.optimizer_D_classification_0.zero_grad()
        self.backward_D_A_classification_0(1 - self.random_number)
        self.backward_D_B_classification_0(1 - self.random_number)
        # gnorm_D_classification_0 = torch.nn.utils.clip_grad_norm(
        #     itertools.chain(
        #         self.netD_A_classification_0.parameters(),
        #         self.netD_B_classification_0.parameters(),
        #     ),
        #     self.opt.max_gnorm,
        # )
        self.optimizer_D_classification_0.step()

        # gnorm_DeConv = torch.nn.utils.clip_grad_norm(
        #     self.netDeConv.parameters(), self.opt.max_gnorm
        # )
        self.optimizer_DeConv.step()
