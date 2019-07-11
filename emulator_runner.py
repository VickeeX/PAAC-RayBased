from multiprocessing import Process


class EmulatorRunner(Process):

    def __init__(self, id, emulator, variable, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulator = emulator
        self.variable = variable
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                self.barrier.put(None)
                break
            # for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
            new_s, reward, episode_over = self.emulator.next(self.variable[-1])
            if episode_over:
                self.variable[0] = self.emulator.get_initial_state()
            else:
                self.variable[0] = new_s
            self.variable[1] = reward
            self.variable[2] = episode_over
            count += 1
            self.barrier.put(True)




