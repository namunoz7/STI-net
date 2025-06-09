import argparse


def parser_gans():
    parser = argparse.ArgumentParser(description="Training cycle gans to obtain Susceptibility Tensor Imaging")
    parser.add_argument("--phase_folder", type=str, default="../../Imagenes/STI/Phase/",
                        help="Folder containing the phase images")
    parser.add_argument("--chi_folder", type=str, default="../../Imagenes/STI/Chi",
                        help="Folder containing the susceptibility tensor images")
    parser.add_argument("--n_train", type=float, default=0.8,
                        help="Percentage of training images obtained from the data")
    parser.add_argument("--n_val", type=float, default=0.2,
                        help="Percentage of validation images obtained from the data")
    parser.add_argument("--device", type=str, default="cpu", help="Number of gpu to use")
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--decay_epochs", type=int, default=100,
                        help="epoch to start linearly decaying the learning rate to 0. (default:100)")
    parser.add_argument("-b", "--batch_size", default=1, type=int,
                        metavar="N",
                        help="batch size (default: 40), this is the total batch size of all GPUs on the"
                             "current node when using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate. (default:0.0002)")
    parser.add_argument("-p", "--print-freq", default=100, type=int,
                        metavar="N", help="print frequency. (default:100)")
    parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
    parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
    parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
    parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
    parser.add_argument("--size", type=int, default=48,
                        help="Initial size of the images. (default:48)")
    parser.add_argument("--outf", default="./Outputs",
                        help="folder to output images. (default:`./Outputs`).")
    parser.add_argument("--manualSeed", type=int,
                        help="Seed for initializing training. (default:none)")
    args = parser.parse_args()
    return args
