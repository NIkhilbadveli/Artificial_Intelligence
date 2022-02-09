# Rescale data (between 0 and 1)
from pandas import read_csv
from numpy import set_printoptions, linalg
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from matplotlib import pyplot as plt

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
mm_scaler = MinMaxScaler(feature_range=(0, 1))
sd_scaler = StandardScaler()
rescaledX = mm_scaler.fit_transform(X)  # Rescales each column between 0 and 1 by taking the max value in that column
rescaledX2 = sd_scaler.fit_transform(X)
# sd_scaler Rescales each column into Gaussian distribution with 0 mean and 1 standard deviation
rescaledX3 = Normalizer().fit_transform(X)
# Normalizer rescales each row such that the norm/length of that row becomes 1
rescaledX4 = Binarizer(threshold=20.0).fit_transform(X)
# Binarizer rescales each value less than or equal to threshold as '0' and the others as '1'

# summarize transformed data
set_printoptions(precision=3)

print(X)
print('\nRescaled values with MinMaxScaler:-')
print(rescaledX)
print('\nRescaled values with Standard Scaler :-')
print(rescaledX2)
print('\nRescaled values with Normalized Scaler :-')
print(rescaledX3)
print('\nRescaled values with Binarizer Scaler :-')
print(rescaledX4)
