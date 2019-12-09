class Config:
    def __init__(self, name):
        # Run metadata
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.batch_size = 32
        self.epochs = 50
        self.validation = True
        self.test = True

        # Networks
        self.generator = None
        self.gen_args = {}

        self.disc_y = None
        self.disc_y_args = {}

        self.disc_z = None
        self.disc_z_args = {}

        self.generator = None
        self.gen_args = {}

        self.encoder = None
        self.enc_args = {}

        self.inferer = None
        self.inf_args = {}

        self.classifier = None
        self.class_args = {}

        # Optimizers
        self.gen_optimizer = None
        self.gen_optimizer_args = None

        self.disc_y_optimizer = None
        self.disc_y_optimizer_args = None

        self.disc_z_optimizer = None
        self.disc_z_optimizer_args = None

        self.enc_optimizer = None
        self.enc_optimizer_args = None

        self.clas_optimizer = None
        self.clas_optimizer_args = None

        self.inf_optimizer = None
        self.inf_optimizer_args = None

        # Data dimensions
        self.channels = None
        self.zshape = None
        self.xshape = None
        self.dataset_size = None
        self.valid_size = None
        self.test_size = None

        # Noise sampling
        self.noise_provider = None
        self.noise_provider_args = {}

        # Metrics
        self.fid_model = None
        self.metrics = {}
