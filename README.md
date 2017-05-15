## Project-Snack-Food
# Project-Snack-Food


##Introduction
For the final project we choose the project C which is a recommendation engines, for this we have to create our own dataset based on websites, or search for an available dataset which represent food items, and rating given to these items by pepole.\\
With no real suitable food dataset available for recommendation system, we choose to build our own dataset, it is represented by the users id, food id and by the rating of the food given by users.\\


##Model KNN
We started by testing the first algorithm that came to mind and which is included in mooc: The K Nearest Neighbor (KNN)
We transform the problem to a simple supervised learning problem that consist on predicting the rating based on the user and the food.
After making a cross validation we decided to take K=3500 (K represent the number od neighbor that the algorithm will take into consideration).
The RMS error is 1.16 for this algorithm.

##Memory Based
For the next steps, we applied Memory Based which is based on the similarity calculated by the different evaluation metrics. For us, we used Cosine similar and citybloc.\\
This measurement is called vector similarity. We can treat two users as vectors in n-dimensional space, where n is the number of items in the database. As with any two vectors, we can compare the angle between them. Intuitively, if the two vectors generally point in the same direction, they get a positive similarity; if they point in opposite directions, they get a negative similarity. To simulate this we just take the cosine the angle between these two vectors, which gives us a value from -1 to 1.

##Model Based
Next, we used Model Based which is based on matrix multiplication or Matrix Factorization (MF): It's an unsupervised learning method which consist on decomposition and  reduction of dimension.\\
The purpose of these algorithms is to find the latent preferences of the users (in the form of a matrix P) and the items (in the form of a Q matrix) from the assigned scores. The multiplication P and Q gives the prediction matrix.
Once it is done, we improve P and Q to have a product that conforms to the matrix of the original notes. Here are the different methods we used: SVD Singular Vector Decomposition: X = U x S x V.T to do it we used the librairy scipy.sparse.linalg. \\
U presents the User-Latent matrix and V.T the Item-Latent matrix NMF Non-Negative Matrix Factorization.

