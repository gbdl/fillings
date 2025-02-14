import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='M3')


    parser.add_argument(
            "--loadname",
            type=str,
            default=None,
            help="name of the experiment to load as a starting point of the evolution. If None will first pre-train",
        )
    

    parser.add_argument(
            "--cpu",
            type=str,
            default="true",
            choices=["true", "false"],
            help="If true forces cpu mode regardless of available gpus, otherwise tries to use a gpu if available",
        )


    # --- Manifold parameters --- #
    parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="Parameter k entering in the choice of Anderson-Dehn filling",
        )


    parser.add_argument(
            "--cutoff",
            type=float,
            default=0.95,
            help="Percentage of the z-range to keep",
        )

    parser.add_argument(
            "--sampling",
            type=str,
            default="volume",
            choices=["uniform", "volume"],
            help="Method used to sample points in the domain",
        )

    parser.add_argument(
            "--conformal_factor",
            type=float,
            default=100,
            help="Solve the equation Ricci = -2*g/conformal_factor",
        )



    # --- Curriculum parameters --- #

    parser.add_argument(
            "--norm_type",
            type=str,
            default="L1",
            choices=["L1", "L2"],
            help="Loss type for the mean relative error",
        )

    parser.add_argument(
        "--resampling",
        type= str, 
        default="importance",
        choices=["none", "importance"],
        help = "Whether to sample more points in places where error is higher"
    )

    parser.add_argument(
        "--resampling_bry",
        type= str, 
        default="importance",
        choices=["none", "importance"],
        help = "Whether to sample more points on the boundary in places where error is higher"
    )

    parser.add_argument(
        "--batch_size_bulk",
        type=int,
        default = 16,
        help = "Bulk batch size, the actual number of points is 8 times this number"
    )


    parser.add_argument(
        "--batch_size_bry",
        type=int,
        default = 32,
        help = "Boundary batch size, the actual number of points is 2*14 times this number"
    )

    parser.add_argument(
        "--batch_size_pretraining",
        type=int,
        default = 256,
        help = "Batch size during pretraining"
    )

    parser.add_argument(
            "--lr_pre",
            type=float,
            default=3e-4,
            help="learning rate pretraining",
        )


    parser.add_argument(
            "--pretraining_uniform",
            type=str,
            default="false",
            choices=["true", "false"],
            help="If true forces the pre-training with uniform_sampling",
        )

    parser.add_argument(
            "--resampling_frequency",
            type= int, 
            default=10,
            help = "After how many iterations resample the importance points"
        )

    parser.add_argument(
            "--resampling_fraction",
            type= float, 
            default=0.6,
            help = "Fraction of points obtained via importance resampling"
        )


    parser.add_argument(
            "--threshold_pretraining",
            type=float,
            default=0.03,
            help="Threshold for stopping the pre-training",
        )


    parser.add_argument("--hweight",
        type=float,
        default=1.,
        help="weight factor for imposing continuity",
        )

    parser.add_argument("--Kweight",
        type=float,
        default=1.,
        help="weight factor for imposing differentiability",
        )

    parser.add_argument("--Rweight",
        type=float,
        default=1.,
        help="weight factor for imposing the Einstein equations",
        )


    # --- Network hyperparameters --- #

    parser.add_argument(
            "--activation",
            type=str,
            default="tanh",
            choices=["id", "gelu", "tanh"],
            help="Activation function in the neural network. 'id' means no activation",
        )

    parser.add_argument("--width",
        type=int,
        default=10,
        help="width of the neural networks",
        )

    parser.add_argument("--depth",
        type=int,
        default=2,
        help="depth of the neural networks",
        )

    # --- Optimizer hyperparameters --- #

    parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd", "ECDSep_scaled" ],
            help="Optimizer used during training",
        )

    parser.add_argument(
            "--load_opt",
            type=str,
            default="false",
            choices=["true", "false"],
            help="load the optimizer states",
        )

    parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="learning rate",
        )

    parser.add_argument(
            "--wd",
            type=float,
            default=0.,
            help="weight decay",
        )


    parser.add_argument(
            "--beta1",
            type=float,
            default=0.9,
            help="beta1 for the adam optimizer",
        )

    parser.add_argument(
            "--beta2",
            type=float,
            default=0.999,
            help="beta2 for the adam optimizer",
        )

    parser.add_argument(
            "--momentum",
            type=float,
            default=0.99,
            help="momentum for the sgd optimizer",
        )


    parser.add_argument(
            "--eta",
            type=float,
            default=1.5,
            help="eta for the ECD optimizers",
        )

    parser.add_argument(
            "--F0",
            type=float,
            default=0.,
            help="F0 for the ECD optimizers",
        )

    parser.add_argument(
            "--nu",
            type=float,
            default=1e-3,
            help="nu for the ECD optimizers",
        )

    parser.add_argument('--name', type=str, default='test', help='experiment name if not using wandb, otherwise it will be the wandb id')


    ## wandb arguments
    parser.add_argument('--use_wandb', type=str, default='false', choices = ["true", "false"], help='Use wandb to track the run')
    parser.add_argument('--project', type=str, default='fillings', help='wandb project')


    parser.add_argument(
            "--debug",
            type=str,
            default="false",
            choices=["true", "false"],
            help="debug mode",
        )


    return parser

