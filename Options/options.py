import argparse


class options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--n_samples', type=int, default=500,
                            help="Number of images multiplied by 2 to be added to each set")
        parser.add_argument('--y_l', type=int,default=15)
        parser.add_argument('--y_u', type=int, default=20)
        parser.add_argument('--mid_l', type=int, default=30)
        parser.add_argument('--mid_u', type=int, default=40)
        parser.add_argument('--o_l', type=int, default=50)
        parser.add_argument('--o_u', type=int, default=60)
        parser.add_argument('--b_size', type=int, default=50)
        parser.add_argument('--epochs', type=int, default=50)
        self.parser = parser.parse_args()