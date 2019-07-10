from multiprocessing import Process
import ray


class EmulatorRunner():
    def __init__(self, id, emulator, variable, signal1, signal2):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulator = emulator
        self.variable = variable
        self.signal1 = signal1
        self.signal2 = signal2

    def run(self):
        count = 0
        while True:
            instruction = self.signal1.get()
            if instruction is None:
                break
            new_s, reward, episode_over = ray.get(self.emulator.next.remote(self.variable))
            if episode_over:
                self.variable[0] = ray.get(self.emulator.get_initial_state())
            else:
                self.variable[0] = new_s
            self.variable[1] = reward
            self.variable[2] = episode_over
            count += 1
            self.signal2.put(True)

            # for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
            #     new_s, reward, episode_over = emulator.next(action)
            #     if episode_over:
            #         self.variables[0][i] = emulator.get_initial_state()
            #     else:
            #         self.variables[0][i] = new_s
            #     self.variables[1][i] = reward
            #     self.variables[2][i] = episode_over
            # count += 1
            # self.barrier.put(True)
