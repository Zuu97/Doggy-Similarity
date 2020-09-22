import os
import joblib
import cv2 as cv
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from variables import*

class DogSimilarity(object):
    def __init__(self):
        self.n_clusters = n_clusters
        self.data_dir = data_dir
        self.visual_words_path = visual_words_path
        self.feature_path = feature_path
        self.train_images_dir= train_images_dir
        self.kmeans_weights = kmeans_weights

    def load_data(self):
        if not os.path.exists(self.train_images_dir):
            print("Training Images Saving")
            images = []
            classes = []
            dog_folders = os.listdir(self.data_dir)
            for label in list(dog_folders):
                label_dir = os.path.join(self.data_dir, label)
                label_images = []
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    images.append(img)
                    classes.append(int(label))
            images = np.array(images)
            classes = np.array(classes)
            np.savez(self.train_images_dir, name1=images, name2=classes)
        else:
            print("Training Images Loaded")
            data = np.load(self.train_images_dir)
            images = data['name1']
            classes = data['name2']

        self.classes, self.images = shuffle(classes, images)

    def sift_extractor(self, img):
        self.extractor = cv.xfeatures2d.SIFT_create()
        keypoints, descriptors = self.extractor.detectAndCompute(img, None)
        return keypoints, descriptors

    def compute_features(self):
        feature_array = []
        descriptor_array = []
        for img in self.images:
            keypoints, descriptors = self.sift_extractor(img)
            descriptor_array.extend(descriptors)
            feature_array.append(descriptors)

        descriptor_array = np.array(descriptor_array)
        feature_array = np.array(feature_array)

        self.scalar = StandardScaler()
        self.scalar.fit(descriptor_array)
        descriptor_array = self.scalar.transform(descriptor_array)
        return [descriptor_array, feature_array]

    # def visual_vocabulary(self):
    #     if not os.path.exists(self.visual_words_path):
    #         print("Visual Vocabulary Creating")
    #         descriptor_array = self.compute_features()[0]
    #         kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
    #         kmeans.fit(descriptor_array)
    #         visual_words = kmeans.cluster_centers_
    #         np.save(self.visual_words_path, visual_words)
    #     else:
    #         print("Visual Vocabulary Loading")
    #         visual_words = np.load(self.visual_words_path)
    #     self.visual_words = visual_words

    # @staticmethod
    # def find_index(v1, v2):
    #     distances = []
    #     for val in v2:
    #         dist = distance.euclidean(v1, val)
    #         distances.append(dist)
    #     return np.argmax(np.array(distances))

    # def create_histogram(self, feature_array):
    #     feature_hist = []
    #     for features in feature_array:
    #         histogram = np.zeros(len(self.visual_words))
    #         for feature in features:
    #             print(DogSimilarity.find_index(feature, self.visual_words))
    #             histogram[DogSimilarity.find_index(feature, self.visual_words)] += 1
    #         feature_hist.append(histogram)
    #     feature_hist = np.array(feature_hist)
    #     return feature_hist

    def visual_vocabulary(self):
        if not os.path.exists(self.kmeans_weights):
            print("Kmeans Model Fitting")
            descriptor_array = self.compute_features()[0]
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            self.kmeans.fit(descriptor_array)
            joblib.dump(self.kmeans, self.kmeans_weights)

        else:
            print("Kmeans Model Loading")
            self.kmeans = joblib.load(self.kmeans_weights)

    def create_histogram(self, feature_array):
        feature_hist = []
        for features in feature_array:
            histogram = np.zeros(n_clusters)
            for feature in features:
                idx = self.kmeans.predict([feature])[0]
                histogram[idx] += 1
            feature_hist.append(histogram)
        feature_hist = np.array(feature_hist)
        return feature_hist

    def normalize_features(self,feature_array):
        norm_feature_array =[]
        for features in feature_array:
            features = self.scalar.transform(features)
            norm_feature_array.append(np.array(features))
        return norm_feature_array

    def feature_histograms(self):
        self.visual_vocabulary()
        if not os.path.exists(self.feature_path):
            print("train features are Creating")
            feature_array = self.compute_features()[1]
            feature_array = self.normalize_features(feature_array)
            self.features = self.create_histogram(feature_array)
            np.save(self.feature_path, self.features)

        else:
            print("train features are Loading")
            self.features = np.load(self.feature_path)

    def predict_neighbour(self):
        input_idx = np.random.choice(len(self.images))
        data = self.features[input_idx]

        neighbor = NearestNeighbors(n_neighbors = 4)
        neighbor.fit(self.features)
        result = neighbor.kneighbors([data])[1].squeeze()
        fig=plt.figure(figsize=(6, 6))
        fig.add_subplot(2, 2, 1)
        plt.title('Input Image')
        plt.imshow(self.images[input_idx])
        print("\nInput image label : {}".format(self.classes[input_idx]))
        for i in range(2, 5):
            neighbour_img_id = result[i-1]
            fig.add_subplot(2, 2, i)
            plt.title('Neighbour {}'.format(i-1))
            plt.imshow(self.images[neighbour_img_id])
            print("Neighbour image {} label : {}".format(i-1, self.classes[neighbour_img_id]))
        plt.show()

model = DogSimilarity()
model.load_data()
model.feature_histograms()
model.predict_neighbour()