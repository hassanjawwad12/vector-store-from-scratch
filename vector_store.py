import numpy as np 

class VectorStore:
    def __init__(self):
        self.vector_data = {} # dictionary to store vectors
        self.vector.index ={} # dictionary to index structure for retrieval 

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store

        Args: 
            vector_id (str, int) : the unique identifier for the vector
            vector (np.array): the vector to be stored
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store

        Args:
            vector_id (str, int): the unique identifier for the vector

        Returns:
            np.array: the vector
        """
        return self.vector_data.get(vector_id)    

    def update_index(self, vector_id, vector):
        # update the index structure
        for i, value in enumerate(vector):
            if i not in self.vector_index:
                self.vector_index[i] = {}
            if value not in self.vector_index[i]:
                self.vector_index[i][value] = set()
            self.vector_index[i][value].add(vector_id)
