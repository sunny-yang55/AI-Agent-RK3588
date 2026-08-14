class ConfidenceFilter:

    def __init__(self):

        self.min_length = 2

    def check(self, text):

        if not text:

            return False

        if len(text) < self.min_length:

            return False

        return True
