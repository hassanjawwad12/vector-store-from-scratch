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
        """
        Update the index structure with the new vector

        Args:
            vector_id (str, int): the unique identifier for the vector
            vector (np.array): the vector to be indexed
        """
        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def get_similar_vectors(self, quer_vector, num_results=5):
        """
        Retrieve the most similar vectors to the query vector

        Args:
            query_vector (np.array): the query vector
            num_results (int): the number of most similar vectors to return

        Returns:
            list: A list of tuples, where each tuple contains the vector_id and similarity score
        """
        similarities = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(quer_vector, vector) / (np.linalg.norm(quer_vector) * np.linalg.norm(vector))
            similarities.append((vector_id, similarity))
        #sort the similarity in descending order    
        similarities.sort(key=lambda x: x[1], reverse=True)
        #return the top num_results
        return similarities[:num_results]
      