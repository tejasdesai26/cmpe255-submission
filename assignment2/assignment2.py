# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class svm_solution:

	def create_pipeline_load_data_print_plots_prf1s_mtrx():
		pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
		svc = SVC(kernel='rbf', class_weight='balanced')
		model = make_pipeline(pca, svc)
		
		faces = fetch_lfw_people(min_faces_per_person=60)
		print('data loaded')
		print(faces.target_names)
		print(faces.images.shape)
		Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)
		
		param_grid = {'svc__C': [1, 10, 20, 50, 100],
				'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.0004]}
		grid = GridSearchCV(model, param_grid)

		#%time grid.fit(Xtrain, ytrain)
		grid.fit(Xtrain, ytrain)
		print(grid.best_params_)
		model = grid.best_estimator_
		yfit = model.predict(Xtest)
		
		fig, ax = plt.subplots(4, 6)
		for i, axi in enumerate(ax.flat):
			axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
			axi.set(xticks=[], yticks=[])
			axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
					color='black' if yfit[i] == ytest[i] else 'red')
		fig.suptitle('Predicted Names', size=14);
		
		print(classification_report(ytest, yfit, target_names=faces.target_names))
		
		mat = confusion_matrix(ytest, yfit)
		sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
				xticklabels=faces.target_names,
				yticklabels=faces.target_names)
		plt.xlabel('true label')
		plt.ylabel('predicted label');

if __name__ == "__main__":
    svm_solution.create_pipeline_load_data_print_plots_prf1s_mtrx()
