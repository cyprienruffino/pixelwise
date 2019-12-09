class Config:
    def __init__(self, name):
        # Run metadata
        self.name = name
        self.seed = None

        # Training settings
        self.batch_size = None
        self.epochs = None
        self.k = None  # Number of D updates vs G updates
        self.lmbda = None
        self.validation = None
        self.test = None

        # Losses
        self.loss_gen = None
        self.loss_disc_fake = None
        self.loss_disc_real = None

        # Optimizers
        self.gen_optimizer = None
        self.gen_optimizer_args = None

        self.disc_optimizer = None
        self.disc_optimizer_args = None

        # Data dimensions
        self.channels = None
        self.nz = None  # Number of channels in Z
        self.zx = None  # Size of each spatial dimensions in Z
        self.npx = None  # (zx * 2^ depth)
        self.dataset_size = None
        self.valid_size = None
        self.test_size = None

        # Networks
        self.generator = None
        self.gen_args = {}

        self.discriminator = None
        self.disc_args = {}

        # Noise sampling
        self.noise_provider = None
        self.noise_provider_args = {}

        # Metrics
        self.fid_model = None
        self.metrics = {}
