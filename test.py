import unittest
from Eigenfaces import *
from aphw7p2 import *

class Test(unittest.TestCase):
    def test1(self):
        X = loadFaces('att_faces')
        self.assertEqual(X.shape, (900, 400))

    def test2(self):
        X = loadFaces('att_faces')
        cov = np.dot(X, np.transpose(X)) / X.shape[1]
        efaces, evals = findEigenFaces(cov, 25)
        self.assertGreaterEqual(evals[0], 190)

    def test3(self):
        X = loadFaces('att_faces')
        cov = np.dot(X, np.transpose(X)) / X.shape[1]
        efaces, evals = findEigenFaces(cov, 25)
        dataset = createDataset('att_faces', efaces)
        self.assertEqual(len(dataset), 400)
        self.assertEqual(type(dataset[0][1]), type('Hi'))

    def test4(self):
        face_vec, face = loadImage('att_faces/s1/3.pgm')
        self.assertEqual(face_vec.shape, (900,1))
        self.assertEqual(face_vec[-2, 0], 31)

    def test5(self):
        X = loadFaces('att_faces')
        cov = np.dot(X, np.transpose(X)) / X.shape[1]
        efaces, evals = findEigenFaces(cov, 25)
        dataset = createDataset('att_faces', efaces)
        face_vec, face = loadImage('att_faces/s1/3.pgm')
        result, klist = kNN(dataset, face_vec, efaces, 5)
        self.assertEqual(len(klist), 5)
        self.assertEqual(result, 's1')

    def test6(self):
        arr = generateArray()
        self.assertEqual(arr.shape, (10, 10))
        self.assertEqual(arr[5, 3], arr[6, 2])

    def test7(self):
        arr, num = play(silent=True)
        self.assertEqual(arr[3, 5], num)


if __name__=='__main__':
    unittest.main()