# -*- coding: utf-8 -*-

"""
    File name    :    test_atari_emulator_remote
    Date         :    08/07/2019
    Description  :    {TODO}
    Author       :    VickeeX
"""
import argparse, numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='pong', help='Name of game', dest='game')
    parser.add_argument('-d', '--device', default='/gpu:0', type=str,
                        help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms',
                        help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg,
                        help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized",
                        dest="visualize")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float,
                        help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float,
                        help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int,
                        help="Nr. of global steps during which the learning rate will be linearly annealed towards zero",
                        dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic)",
                        dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float,
                        help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm",
                        dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global",
                        help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)",
                        dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps",
                        dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int,
                        help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    parser.add_argument('--arch', default='NIPS',
                        help="Which network architecture to use: from the NIPS or NATURE paper", dest="arch")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg,
                        help="If True, training episodes will be terminated when a life is lost (for games)",
                        dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=4, type=int,
                        help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int,
                        help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg,
                        help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()

    from atari_emulator import AtariEmulator
    from ale_python_interface import ALEInterface

    filename = args.rom_path + "/" + args.game + ".bin"
    ale_int = ALEInterface()
    ale_int.loadROM(str.encode(filename))
    num_actions = len(ale_int.getMinimalActionSet())
    create_environment = lambda i: AtariEmulator.remote(i, args)

    emulators = np.asarray([create_environment(i) for i in range(4)])
    variables = [(np.asarray([emulator.get_initial_state.remote() for emulator in emulators], dtype=np.uint8)),
                 (np.zeros(4, dtype=np.float32)),
                 (np.asarray([False] * 4, dtype=np.float32)),
                 (np.zeros((4, num_actions), dtype=np.float32))]

    for step in range(10):
        for i, (emulator, action) in enumerate(zip(emulators, variables[-1])):
            new_s, reward, episode_over = emulator.next.remote(action)
            if episode_over:
                variables[0][i] = emulator.get_initial_state.remote()
            else:
                variables[0][i] = new_s
            variables[1][i] = reward
            variables[2][i] = episode_over
        print("get batch data %d." % i)
