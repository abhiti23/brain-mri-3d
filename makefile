visualize_slice: artifacts/coefficients_final.npz src/pipeline/visualize_fd_step.py
	echo ">>> Visualizing slices for a random subject"
	python src/pipeline/visualize_fd_step.py

train_nn: src/analysis/train_nn.py artifacts/coefficients_final.npz
	echo ">>> training model from functional coefficients"
	python src/analysis/train_nn.py

