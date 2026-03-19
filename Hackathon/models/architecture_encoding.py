class Architecture:

    def __init__(self, depth, hidden_units, activation, dropout):

        self.depth = depth
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout

    def __repr__(self):

        return f"Architecture(depth={self.depth}, layers={self.hidden_units}, activation={self.activation}, dropout={self.dropout})"