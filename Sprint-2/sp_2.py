
# FEATURE EXTRACTION USING COUNT VECTORISATION

from sklearn.feature_extraction.text import CountVectorizer

x = np.array(data["Text"])
y = np.array(data["language"])
x = [str(text) for text in x]

cv = CountVectorizer()
X = cv.fit_transform(x)

X_array = X.toarray()

X.shape

"""This means that your count vectorized data consists of 22,000 rows (documents) and 278,217 columns (unique words in your vocabulary)"""

word_frequencies = X_array.sum(axis=0)
print(word_frequencies)

"""The first value (4) represents the total count of the first word in the vocabulary.The second value (1) represents the total count of the second word in your vocabulary."""
