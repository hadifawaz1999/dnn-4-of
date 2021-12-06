import parameters as params
import transmitor as tx
import modulator as md
import channel
import projectRunner as prj
import numpy as np


if __name__ == "__main__" :
    # Initialize parameters
    parameter = params.Parameters()
    # Initialize the Transmitor
    transmitor = tx.Transmitor()
    # Initialize the Transmitor
    modulator = md.Modulator()
    # Initialize the Channel
    channel = channel.Channel()
    # Initialize projectRunner
    runner = prj.projectRunner(parameter, transmitor, modulator, channel)
    # run the project
    runner.run()