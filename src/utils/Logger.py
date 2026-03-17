import sys


class CleanDualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        # 1. Scrive TUTTO sul terminale (incluse le barre tqdm)
        self.terminal.write(message)

        # 2. Scrive sul file SOLO se non è un'animazione di tqdm (niente \r)
        if '\r' not in message:
            self.log.write(message)
            self.log.flush()  # Salva istantaneamente su disco!

    def flush(self):
        self.terminal.flush()
        self.log.flush()

