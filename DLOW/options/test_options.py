from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        # self.parser.add_argument('--results_dir', type=str, default='/home/data/fwl/PancreasTumor/data/data_transfer/',
        #                          help='saves results here.')
        self.parser.add_argument('--results_dir', type=str, default='./results/',
                                 help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--rand', type=float, default=0.5, help='rand dirk add')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=12940, help='how many test images to run')
        self.parser.add_argument('--label_intensity_styletransfer', nargs='+', type=float, default=1, help='style transfer')
        self.isTrain = False  # 刚到手的源代码，self.isTrain是True， 但是testmodel要false才能运行，所以改成false了

