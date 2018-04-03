class Config:
    def __init__(self, name):
        # Run metadata
        self.name = name
        self.seed = None

        # Training settings
        self.batch_size = None
        self.epoch_iters = None
        self.epochs = None
        self.k = None  # Number of D updates vs G updates

        # Data dimensions
        self.convdims = None  # 2D or 3D convolutions
        self.nz = None  # Number of channels in Z
        self.zx = None  # Size of each spatial dimensions in Z
        self.zx_sample = None
        self.npx = None  # (zx * 2^ depth)

        # Networks
        self.generator = None
        self.gen_args = {}

        self.discriminator = None
        self.disc_args = {}

        # Losses
        self.loss_disc_fake = None
        self.loss_disc_true = None
        self.loss_gen = None

        # Optimizers
        self.gen_optimizer = None
        self.gen_optimizer_args = None

        self.disc_optimizer = None
        self.disc_optimizer_args = None

        # Data providers
        self.disc_data_provider = None
        self.disc_data_provider_args = {}

        self.gen_data_provider = None
        self.gen_data_provider_args = {}


